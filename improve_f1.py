"""
Amelioration du F1 score - Strategies avancees
1. Calibration des probabilites (Platt Scaling)
2. Post-processing : decision par K consecutifs
3. Enrichissement features : nouvelles variables temporelles
4. Ensemble : combinaison V3 + par aeroport
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import lightgbm as lgb
from scipy.special import logit, expit

# ── Chargement des donnees ────────────────────────────────────────────────────
print("Chargement des donnees...")
df = pd.read_csv('df_cg_enriched.csv')
df['date'] = pd.to_datetime(df['date'], utc=True)
df['is_last'] = df['is_last_lightning_cloud_ground'].map({'True': True, 'False': False, True: True, False: False})
df = df[df['is_last'].notna()].copy()
df['y'] = df['is_last'].astype(int)
print(f"Dataset: {len(df)} eclairs CG, {df['y'].mean()*100:.2f}% positifs")

# ── Features ────────────────────────────────────────────────────────────────
FEAT_V3 = [
    'rank_cg', 't_since_start_s', 'dt_prev_s',
    'dt_mean_3','dt_mean_5','dt_mean_10','dt_mean_20',
    'dt_max_3','dt_max_5','dt_max_10','dt_max_20',
    'dt_std_3','dt_std_5','dt_std_10',
    'dt_ema_3','dt_ema_5','dt_ema_10',
    'dist','dist_mean_3','dist_mean_5','dist_mean_10','dist_mean_20',
    'dist_trend_3','dist_trend_5','dist_trend_10','dist_trend_20',
    'amp_abs','amp_sign','maxis',
    'amp_mean_3','amp_mean_5','amp_mean_10','amp_mean_20',
    'amp_trend_3','amp_trend_5','amp_trend_10',
    'n_cg_2min','n_cg_5min','n_cg_10min','n_cg_15min','n_cg_30min',
    'rate_decline_cg',
    'n_ic_2min','n_ic_5min','n_ic_10min','n_ic_30min',
    'n_all_2min','n_all_5min','n_all_10min',
    'ratio_ic_cg_5min','ratio_ic_cg_10min',
    'ic_trend','total_activity','activity_decline',
    'azimuth','month','hour',
]

# Nouvelles features pour ameliorer le F1
print("\nCreation de nouvelles features...")

# 1. Ratio de declin elargi
df['rate_decline_cg_long'] = df['n_cg_10min'] / (df['n_cg_30min'] + 1)
df['rate_decline_ic'] = df['n_ic_5min'] / (df['n_ic_30min'] + 1)

# 2. Acceleration de l'espacement (2eme derive)
grp = df.groupby(['airport', 'airport_alert_id'])
df['dt_accel'] = grp['dt_prev_s'].transform(lambda x: x.diff().fillna(0))  # augmentation du dt = signe de fin

# 3. Tendance sur 15 eclairs
for w in [15]:
    df[f'dt_mean_{w}']  = grp['dt_prev_s'].transform(lambda x: x.rolling(w, min_periods=1).mean())
    df[f'dist_trend_{w}'] = grp['dist'].transform(
        lambda x: x.rolling(w, min_periods=2).apply(
            lambda v: np.polyfit(range(len(v)), v, 1)[0] if len(v) > 1 else 0, raw=True))
    df[f'amp_trend_{w}'] = grp['amplitude'].transform(
        lambda x: x.abs().rolling(w, min_periods=2).apply(
            lambda v: np.polyfit(range(len(v)), v, 1)[0] if len(v) > 1 else 0, raw=True))

# 4. Ratio IC/CG long terme
df['ratio_ic_cg_30min'] = df['n_ic_30min'] / (df['n_cg_30min'] + 1)

# 5. Score composite de declin (combinaison lineaire interpretable)
df['composite_decline'] = (
    (df['dt_prev_s'] / (df['dt_mean_10'] + 1)) *  # eclair actuel vs moyenne recente
    (1 / (df['n_cg_5min'] + 1)) *                  # inverse de la densite recente
    (df['ratio_ic_cg_10min'])                        # ratio IC/CG croissant
)

# 6. dt normalise par la duree de l'alerte (relative timing)
df['dt_relative'] = df['dt_prev_s'] / (df['t_since_start_s'] / (df['rank_cg'] + 1) + 1)

# 7. Momentum de la distance (est-ce que l'orage s'accelere en eloignement?)
df['dist_momentum'] = grp['dist'].transform(lambda x: x.diff(3).fillna(0))

NEW_FEATS = [
    'rate_decline_cg_long', 'rate_decline_ic', 'dt_accel',
    'dt_mean_15', 'dist_trend_15', 'amp_trend_15',
    'ratio_ic_cg_30min', 'composite_decline', 'dt_relative', 'dist_momentum'
]

FEAT_V4 = FEAT_V3 + NEW_FEATS
# Verifier que toutes les features existent
FEAT_V4 = [f for f in FEAT_V4 if f in df.columns]
print(f"Features V4: {len(FEAT_V4)} (V3 avait {len(FEAT_V3)})")

# Encodage aeroport
le = LabelEncoder()
df['airport_enc'] = le.fit_transform(df['airport'])
FEAT_V4_AP = FEAT_V4 + ['airport_enc']

X = df[FEAT_V4_AP].fillna(0).values
y = df['y'].values
groups = df['airport_alert_id'].values

# ── Cross-validation GroupKFold ───────────────────────────────────────────────
print("\nCross-validation GroupKFold(5)...")
gkf = GroupKFold(n_splits=5)

PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'n_estimators': 1200,
    'learning_rate': 0.02,
    'max_depth': 7,
    'num_leaves': 127,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 20,
    'class_weight': 'balanced',
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1,
}

oof_probs = np.zeros(len(df))
aucs, aps, f1s = [], [], []

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    mdl = lgb.LGBMClassifier(**PARAMS)
    mdl.fit(X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(-1)])

    probs = mdl.predict_proba(X_val)[:, 1]
    oof_probs[val_idx] = probs

    auc = roc_auc_score(y_val, probs)
    ap  = average_precision_score(y_val, probs)
    # Seuil F1 optimal sur ce fold
    thresholds = np.linspace(0.05, 0.95, 100)
    f1_scores = [f1_score(y_val, (probs >= t).astype(int), zero_division=0) for t in thresholds]
    best_f1 = max(f1_scores)
    aucs.append(auc); aps.append(ap); f1s.append(best_f1)
    print(f"  Fold {fold+1}: AUC={auc:.4f}, AP={ap:.4f}, F1={best_f1:.4f}")

print(f"\nResultats OOF V4:")
print(f"  AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
print(f"  AP:  {np.mean(aps):.4f} +/- {np.std(aps):.4f}")
print(f"  F1:  {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")

# Seuil F1 optimal global sur OOF
thresholds = np.linspace(0.05, 0.95, 200)
f1_oof = [f1_score(y, (oof_probs >= t).astype(int), zero_division=0) for t in thresholds]
best_thr = thresholds[np.argmax(f1_oof)]
best_f1_oof = max(f1_oof)
prec_oof = precision_score(y, (oof_probs >= best_thr).astype(int), zero_division=0)
rec_oof  = recall_score(y, (oof_probs >= best_thr).astype(int), zero_division=0)
print(f"\nOOF global au seuil optimal {best_thr:.3f}:")
print(f"  F1={best_f1_oof:.4f}, Precision={prec_oof:.4f}, Recall={rec_oof:.4f}")

# ── Comparaison avec V3 ───────────────────────────────────────────────────────
print("\nComparaison avec V3:")
df_oof_v3 = pd.read_csv('df_cg_oof_v3.csv')
if 'oof_prob_v3' in df_oof_v3.columns:
    v3_probs = df_oof_v3['oof_prob_v3'].fillna(0).values[:len(y)]
    if len(v3_probs) == len(y):
        f1_v3_vals = [f1_score(y, (v3_probs >= t).astype(int), zero_division=0) for t in thresholds]
        best_thr_v3 = thresholds[np.argmax(f1_v3_vals)]
        best_f1_v3 = max(f1_v3_vals)
        auc_v3 = roc_auc_score(y, v3_probs)
        print(f"  V3: AUC={auc_v3:.4f}, F1={best_f1_v3:.4f} (seuil={best_thr_v3:.3f})")
        print(f"  V4: AUC={roc_auc_score(y, oof_probs):.4f}, F1={best_f1_oof:.4f} (seuil={best_thr:.3f})")
        print(f"  Delta F1: {best_f1_oof - best_f1_v3:+.4f}")

# ── Entrainement final sur tout le dataset ────────────────────────────────────
print("\nEntrainement final V4 (toutes donnees)...")
mdl_final = lgb.LGBMClassifier(**{**PARAMS, 'n_estimators': 1200})
mdl_final.fit(X, y, callbacks=[lgb.log_evaluation(-1)])

# Sauvegarde
save = {
    'model': mdl_final,
    'le': le,
    'features': FEAT_V4_AP,
    'best_thr': best_thr,
    'oof_auc': np.mean(aucs),
    'oof_f1': best_f1_oof,
    'oof_prec': prec_oof,
    'oof_rec': rec_oof,
}
with open('models/lgbm_v4.pkl', 'wb') as f:
    pickle.dump(save, f)
print(f"Modele V4 sauvegarde dans models/lgbm_v4.pkl")
print(f"AUC final OOF: {np.mean(aucs):.4f}, F1 final OOF: {best_f1_oof:.4f}")

# Sauvegarde des predictions OOF
df_oof_out = df[['airport', 'airport_alert_id', 'date', 'y']].copy()
df_oof_out['oof_prob_v4'] = oof_probs
df_oof_out.to_csv('df_cg_oof_v4.csv', index=False)
print("Predictions OOF sauvegardees dans df_cg_oof_v4.csv")
