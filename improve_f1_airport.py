"""
Amelioration du F1 - Modeles par aeroport V4 avec features enrichies
+ Evaluation operationnelle sur le set d'eval
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              roc_auc_score, average_precision_score)
import lightgbm as lgb

# ── Chargement des donnees enrichies ─────────────────────────────────────────
print("Chargement...")
df = pd.read_csv('df_cg_enriched.csv')
df['date'] = pd.to_datetime(df['date'], utc=True)
df['is_last'] = df['is_last_lightning_cloud_ground'].map(
    {'True': True, 'False': False, True: True, False: False})
df = df[df['is_last'].notna()].copy()
df['y'] = df['is_last'].astype(int)

grp = df.groupby(['airport', 'airport_alert_id'])

# ── Nouvelles features ────────────────────────────────────────────────────────
print("Feature engineering V4...")
df['rate_decline_cg_long'] = df['n_cg_10min'] / (df['n_cg_30min'] + 1)
df['rate_decline_ic']      = df['n_ic_5min']  / (df['n_ic_30min'] + 1)
df['dt_accel']             = grp['dt_prev_s'].transform(lambda x: x.diff().fillna(0))
df['ratio_ic_cg_30min']    = df['n_ic_30min'] / (df['n_cg_30min'] + 1)
df['composite_decline']    = (
    (df['dt_prev_s'] / (df['dt_mean_10'] + 1)) *
    (1 / (df['n_cg_5min'] + 1)) *
    df['ratio_ic_cg_10min']
)
df['dt_relative']   = df['dt_prev_s'] / (df['t_since_start_s'] / (df['rank_cg'] + 1) + 1)
df['dist_momentum'] = grp['dist'].transform(lambda x: x.diff(3).fillna(0))

for w in [15]:
    df[f'dt_mean_{w}']    = grp['dt_prev_s'].transform(lambda x: x.rolling(w, min_periods=1).mean())
    df[f'dist_trend_{w}'] = grp['dist'].transform(
        lambda x: x.rolling(w, min_periods=2).apply(
            lambda v: np.polyfit(range(len(v)), v, 1)[0] if len(v) > 1 else 0, raw=True))
    df[f'amp_trend_{w}']  = grp['amplitude'].transform(
        lambda x: x.abs().rolling(w, min_periods=2).apply(
            lambda v: np.polyfit(range(len(v)), v, 1)[0] if len(v) > 1 else 0, raw=True))

FEAT_AP = [
    'rank_cg', 't_since_start_s', 'dt_prev_s',
    'dt_mean_3','dt_mean_5','dt_mean_10','dt_mean_15','dt_mean_20',
    'dt_max_3','dt_max_5','dt_max_10','dt_max_20',
    'dt_std_3','dt_std_5','dt_std_10',
    'dt_ema_3','dt_ema_5','dt_ema_10',
    'dist','dist_mean_3','dist_mean_5','dist_mean_10','dist_mean_20',
    'dist_trend_3','dist_trend_5','dist_trend_10','dist_trend_15','dist_trend_20',
    'amp_abs','amp_sign','maxis',
    'amp_mean_3','amp_mean_5','amp_mean_10','amp_mean_20',
    'amp_trend_3','amp_trend_5','amp_trend_10','amp_trend_15',
    'n_cg_2min','n_cg_5min','n_cg_10min','n_cg_15min','n_cg_30min',
    'rate_decline_cg', 'rate_decline_cg_long', 'rate_decline_ic',
    'n_ic_2min','n_ic_5min','n_ic_10min','n_ic_30min',
    'n_all_2min','n_all_5min','n_all_10min',
    'ratio_ic_cg_5min','ratio_ic_cg_10min','ratio_ic_cg_30min',
    'ic_trend','total_activity','activity_decline',
    'dt_accel', 'composite_decline', 'dt_relative', 'dist_momentum',
    'azimuth','month','hour',
]
FEAT_AP = [f for f in FEAT_AP if f in df.columns]
print(f"Features: {len(FEAT_AP)}")

PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'n_estimators': 1200,
    'learning_rate': 0.02,
    'max_depth': 7,
    'num_leaves': 127,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 15,
    'class_weight': 'balanced',
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1,
}

# ── Entrainement et eval par aeroport ─────────────────────────────────────────
airports = sorted(df['airport'].unique())
airport_models = {}
all_results = []

for airport in airports:
    dfa = df[df['airport'] == airport].copy()
    X   = dfa[FEAT_AP].fillna(0).values
    y   = dfa['y'].values
    groups = dfa['airport_alert_id'].values

    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(dfa))
    aucs_f, f1s_f = [], []

    for fold, (tr, val) in enumerate(gkf.split(X, y, groups)):
        mdl = lgb.LGBMClassifier(**PARAMS)
        mdl.fit(X[tr], y[tr],
                eval_set=[(X[val], y[val])],
                callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(-1)])
        probs = mdl.predict_proba(X[val])[:, 1]
        oof[val] = probs
        aucs_f.append(roc_auc_score(y[val], probs))

    # Seuil optimal sur OOF
    thresholds = np.linspace(0.05, 0.95, 200)
    f1_vals = [f1_score(y, (oof >= t).astype(int), zero_division=0) for t in thresholds]
    best_thr = thresholds[np.argmax(f1_vals)]
    best_f1  = max(f1_vals)
    best_auc = np.mean(aucs_f)
    best_prec = precision_score(y, (oof >= best_thr).astype(int), zero_division=0)
    best_rec  = recall_score(y, (oof >= best_thr).astype(int), zero_division=0)
    best_ap   = average_precision_score(y, oof)

    print(f"{airport:10s}: AUC={best_auc:.4f}, AP={best_ap:.4f}, "
          f"F1={best_f1:.4f} (P={best_prec:.3f}, R={best_rec:.3f}) @ thr={best_thr:.3f}")
    all_results.append({'airport': airport, 'auc': best_auc, 'ap': best_ap,
                        'f1': best_f1, 'prec': best_prec, 'rec': best_rec, 'thr': best_thr})

    # Modele final sur toutes les donnees de l'aeroport
    mdl_final = lgb.LGBMClassifier(**{**PARAMS, 'n_estimators': 1200})
    mdl_final.fit(X, y, callbacks=[lgb.log_evaluation(-1)])
    airport_models[airport] = {'model': mdl_final, 'best_thr': best_thr, 'oof': oof}

# ── Synthese ────────────────────────────────────────────────────────────────
print("\n--- Synthese V4 par aeroport ---")
res_df = pd.DataFrame(all_results)
print(f"AUC moyen   : {res_df['auc'].mean():.4f}")
print(f"AP moyen    : {res_df['ap'].mean():.4f}")
print(f"F1 moyen    : {res_df['f1'].mean():.4f}")
print(f"Precision   : {res_df['prec'].mean():.4f}")
print(f"Recall      : {res_df['rec'].mean():.4f}")

# ── Sauvegarde ─────────────────────────────────────────────────────────────
le = LabelEncoder()
le.fit(airports)
save = {
    'airport_models': airport_models,
    'le': le,
    'features': FEAT_AP,
    'results': res_df.to_dict('records'),
}
with open('models/lgbm_per_airport_v4.pkl', 'wb') as f:
    pickle.dump(save, f)
print("\nModele sauvegarde : models/lgbm_per_airport_v4.pkl")

# ── Evaluation sur le set eval ───────────────────────────────────────────────
print("\nGeneration des predictions sur l'eval...")
df_eval = pd.read_csv('segment_alerts_all_airports_eval.csv')
df_eval['date'] = pd.to_datetime(df_eval['date'], utc=True)
df_eval['icloud'] = df_eval['icloud'].map({'True': True, 'False': False}).fillna(df_eval['icloud']).astype(bool)
df_eval_labeled = df_eval[df_eval['airport_alert_id'].notnull()].copy()
df_cg_eval = df_eval_labeled[~df_eval_labeled['icloud']].copy()
df_cg_eval = df_cg_eval.sort_values(['airport', 'airport_alert_id', 'date']).reset_index(drop=True)

# Feature engineering sur eval (meme pipeline)
def compute_features_eval(df_cg, df_full):
    from scipy.stats import linregress
    df = df_cg.copy()
    grp2 = df.groupby(['airport', 'airport_alert_id'])
    df['rank_cg']         = grp2.cumcount()
    df['t_since_start_s'] = (df['date'] - grp2['date'].transform('min')).dt.total_seconds()
    df['dt_prev_s']       = grp2['date'].diff().dt.total_seconds().fillna(0)
    df['amp_abs']         = df['amplitude'].abs()
    df['amp_sign']        = np.sign(df['amplitude'])
    df['month']           = df['date'].dt.month
    df['hour']            = df['date'].dt.hour

    for w in [3, 5, 10, 15, 20]:
        df[f'dt_mean_{w}']    = grp2['dt_prev_s'].transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f'dt_max_{w}']     = grp2['dt_prev_s'].transform(lambda x: x.rolling(w, min_periods=1).max())
        df[f'dt_std_{w}']     = grp2['dt_prev_s'].transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0))
        df[f'dt_ema_{w}']     = grp2['dt_prev_s'].transform(lambda x: x.ewm(span=w, adjust=False).mean())
        df[f'dist_mean_{w}']  = grp2['dist'].transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f'dist_trend_{w}'] = grp2['dist'].transform(
            lambda x: x.rolling(w, min_periods=2).apply(
                lambda v: np.polyfit(range(len(v)), v, 1)[0] if len(v) > 1 else 0, raw=True))
        df[f'amp_mean_{w}']   = grp2['amplitude'].transform(lambda x: x.abs().rolling(w, min_periods=1).mean())
        df[f'amp_trend_{w}']  = grp2['amplitude'].transform(
            lambda x: x.abs().rolling(w, min_periods=2).apply(
                lambda v: np.polyfit(range(len(v)), v, 1)[0] if len(v) > 1 else 0, raw=True))

    def rtc(dates_v, wns):
        res = np.zeros(len(dates_v))
        for i in range(len(dates_v)):
            res[i] = np.sum(dates_v[:i+1] >= dates_v[i] - wns)
        return res

    for wm, wns in [(2, np.timedelta64(120*10**9,'ns')), (5, np.timedelta64(300*10**9,'ns')),
                    (10, np.timedelta64(600*10**9,'ns')), (15, np.timedelta64(900*10**9,'ns')),
                    (30, np.timedelta64(1800*10**9,'ns'))]:
        col = np.zeros(len(df))
        for (ap, aid), idx in grp2.groups.items():
            dv = df.loc[idx, 'date'].values
            col[idx] = rtc(dv, wns)
        df[f'n_cg_{wm}min'] = col

    df['rate_decline_cg'] = df['n_cg_5min'] / (df['n_cg_10min'] + 1)

    for c in ['n_ic_2min','n_ic_5min','n_ic_10min','n_ic_30min',
              'n_all_2min','n_all_5min','n_all_10min']:
        df[c] = 0.0

    if df_full is not None:
        for airport in df['airport'].unique():
            mask = df['airport'] == airport
            ic_d  = np.sort(df_full[(df_full['airport']==airport) & df_full['icloud']]['date'].values)
            all_d = np.sort(df_full[df_full['airport']==airport]['date'].values)
            cg_d  = df.loc[mask, 'date'].values
            for wns, col in [(np.timedelta64(120*10**9,'ns'),'n_ic_2min'),
                              (np.timedelta64(300*10**9,'ns'),'n_ic_5min'),
                              (np.timedelta64(600*10**9,'ns'),'n_ic_10min'),
                              (np.timedelta64(1800*10**9,'ns'),'n_ic_30min')]:
                df.loc[mask, col] = [
                    np.searchsorted(ic_d, t+np.timedelta64(1,'ns')) -
                    np.searchsorted(ic_d, t - wns) for t in cg_d]
            for wns, col in [(np.timedelta64(120*10**9,'ns'),'n_all_2min'),
                              (np.timedelta64(300*10**9,'ns'),'n_all_5min'),
                              (np.timedelta64(600*10**9,'ns'),'n_all_10min')]:
                df.loc[mask, col] = [
                    np.searchsorted(all_d, t+np.timedelta64(1,'ns')) -
                    np.searchsorted(all_d, t - wns) for t in cg_d]

    df['ratio_ic_cg_5min']   = df['n_ic_5min']  / (df['n_cg_5min']  + 1)
    df['ratio_ic_cg_10min']  = df['n_ic_10min'] / (df['n_cg_10min'] + 1)
    df['ratio_ic_cg_30min']  = df['n_ic_30min'] / (df['n_cg_30min'] + 1)
    df['ic_trend']           = df['n_ic_5min'] - df['n_ic_10min'] / 2
    df['total_activity']     = df['n_all_5min']
    df['activity_decline']   = df['n_all_5min'] / (df['n_all_10min'] + 1)
    df['rate_decline_cg_long'] = df['n_cg_10min'] / (df['n_cg_30min'] + 1)
    df['rate_decline_ic']    = df['n_ic_5min']  / (df['n_ic_30min'] + 1)
    df['dt_accel']           = grp2['dt_prev_s'].transform(lambda x: x.diff().fillna(0))
    df['composite_decline']  = (df['dt_prev_s']/(df['dt_mean_10']+1)) * (1/(df['n_cg_5min']+1)) * df['ratio_ic_cg_10min']
    df['dt_relative']        = df['dt_prev_s'] / (df['t_since_start_s'] / (df['rank_cg'] + 1) + 1)
    df['dist_momentum']      = grp2['dist'].transform(lambda x: x.diff(3).fillna(0))
    return df

print("Feature engineering sur eval (peut prendre quelques minutes)...")
df_cg_eval_feat = compute_features_eval(df_cg_eval, df_eval)

# Predictions
rows = []
for airport in airports:
    sub = df_cg_eval_feat[df_cg_eval_feat['airport'] == airport].copy()
    if len(sub) == 0:
        continue
    mdl = airport_models[airport]['model']
    thr = airport_models[airport]['best_thr']
    feats = [f for f in FEAT_AP if f in sub.columns]
    for m in FEAT_AP:
        if m not in sub.columns:
            sub[m] = 0.0
    scores = mdl.predict_proba(sub[FEAT_AP].fillna(0).astype(float))[:, 1]
    sub = sub.copy()
    sub['confidence'] = scores

    for aid, grp_a in sub.groupby('airport_alert_id'):
        grp_s = grp_a.sort_values('date')
        for _, eclair in grp_s.iterrows():
            rows.append({
                'airport': airport,
                'airport_alert_id': int(aid),
                'prediction_date': eclair['date'],
                'predicted_date_end_alert': eclair['date'],
                'confidence': eclair['confidence'],
            })

pred_df = pd.DataFrame(rows)
pred_df.to_csv('predictions_eval_v4.csv', index=False)
print(f"Predictions sauvegardees: {len(pred_df)} lignes")

print("\nTermine.")
