"""
Data Battle 2026 - Comparaison systematique de modeles
=====================================================
Compare LightGBM, XGBoost, Random Forest, Logistic Regression
avec differentes strategies de gestion du desequilibre de classes.

Metriques : AUC-ROC, AUC-PR, F1, Precision, Rappel, MCC
Validation : GroupKFold(5) par alerte
"""

import pandas as pd
import numpy as np
import pickle
import json
import warnings
import time
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, matthews_corrcoef,
    classification_report, precision_recall_curve
)
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ── Chargement des donnees ────────────────────────────────────────────────────
print("=" * 70)
print("CHARGEMENT DES DONNEES")
print("=" * 70)

df = pd.read_csv('df_cg_enriched.csv')
df['date'] = pd.to_datetime(df['date'], utc=True)
y = df['is_last_lightning_cloud_ground'].astype(int)
groups = df['airport_alert_id']

print(f"Dataset : {len(df)} lignes, {df.shape[1]} colonnes")
print(f"Classe positive : {y.sum()} ({y.mean():.2%})")
print(f"Ratio desequilibre : 1:{int((1-y.mean())/y.mean())}")
print()

# ── Features causales ─────────────────────────────────────────────────────────
LEAK_COLS = ['total_in_alert', 'rel_pos', 'remaining_in_alert', 'is_last',
             'confidence', 'rank_in_alert']
META_COLS = ['lightning_id', 'lightning_airport_id', 'date', 'lon', 'lat',
             'icloud', 'airport', 'airport_alert_id',
             'is_last_lightning_cloud_ground']

# Encoder l'aeroport
le = LabelEncoder()
df['airport_enc'] = le.fit_transform(df['airport'])

FEAT = [c for c in df.columns if c not in META_COLS + LEAK_COLS]
# S'assurer que les features de base brutes sont incluses
for c in ['dist', 'azimuth', 'amplitude', 'maxis']:
    if c not in FEAT:
        FEAT.append(c)

# Remove any remaining non-numeric or problematic cols
X = df[FEAT].copy()
for c in X.columns:
    X[c] = pd.to_numeric(X[c], errors='coerce')
X = X.fillna(0)

print(f"Features utilisees : {len(FEAT)}")
print(f"Top 10 : {FEAT[:10]}")
print()

# ── Configuration GroupKFold ──────────────────────────────────────────────────
gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(X, y, groups))

# ── Fonctions utilitaires ─────────────────────────────────────────────────────
def evaluate_oof(y_true, y_proba, name, find_best_threshold=True):
    """Calcule toutes les metriques OOF."""
    auc_roc = roc_auc_score(y_true, y_proba)
    auc_pr = average_precision_score(y_true, y_proba)

    # Trouver le meilleur seuil pour F1
    if find_best_threshold:
        prec, rec, thresholds = precision_recall_curve(y_true, y_proba)
        f1s = 2 * prec * rec / (prec + rec + 1e-8)
        best_idx = np.argmax(f1s)
        best_thr = thresholds[min(best_idx, len(thresholds)-1)]
    else:
        best_thr = 0.5

    y_pred = (y_proba >= best_thr).astype(int)
    f1 = f1_score(y_true, y_pred)
    prec_val = precision_score(y_true, y_pred)
    rec_val = recall_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    print(f"  {name:40s} | AUC={auc_roc:.4f} | AUC-PR={auc_pr:.4f} | "
          f"F1={f1:.4f} | P={prec_val:.3f} | R={rec_val:.3f} | MCC={mcc:.3f} | thr={best_thr:.3f}")

    return {
        'name': name,
        'auc_roc': round(auc_roc, 4),
        'auc_pr': round(auc_pr, 4),
        'f1': round(f1, 4),
        'precision': round(prec_val, 4),
        'recall': round(rec_val, 4),
        'mcc': round(mcc, 4),
        'best_threshold': round(best_thr, 4),
    }


def train_oof(model_fn, name, use_smote=False, smote_strategy=None):
    """Entraine un modele en OOF avec GroupKFold."""
    oof_proba = np.zeros(len(y))
    t0 = time.time()

    for fold_idx, (tr, va) in enumerate(folds):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]

        if use_smote and smote_strategy:
            try:
                sm = smote_strategy(random_state=42, n_jobs=-1) if hasattr(smote_strategy, 'n_jobs') else smote_strategy(random_state=42)
                X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)
            except Exception:
                sm = smote_strategy(random_state=42)
                X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)
        else:
            X_tr_res, y_tr_res = X_tr, y_tr

        model = model_fn()
        model.fit(X_tr_res, y_tr_res)

        if hasattr(model, 'predict_proba'):
            oof_proba[va] = model.predict_proba(X_va)[:, 1]
        else:
            oof_proba[va] = model.decision_function(X_va)

    elapsed = time.time() - t0
    result = evaluate_oof(y, oof_proba, name)
    result['time_s'] = round(elapsed, 1)
    return result, oof_proba


# ══════════════════════════════════════════════════════════════════════════════
# COMPARAISON DES MODELES
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("COMPARAISON DES MODELES")
print("=" * 70)
print(f"{'Modele':40s} | {'AUC':>6s} | {'AUC-PR':>6s} | {'F1':>6s} | {'P':>5s} | {'R':>5s} | {'MCC':>5s} | {'thr':>5s}")
print("-" * 110)

results = []
oof_dict = {}

# ── 1. LightGBM baseline (class_weight balanced) ─────────────────────────────
def lgbm_balanced():
    return lgb.LGBMClassifier(
        n_estimators=1200, learning_rate=0.02, max_depth=7,
        num_leaves=127, subsample=0.8, colsample_bytree=0.8,
        class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1)

r, oof = train_oof(lgbm_balanced, "LightGBM balanced")
results.append(r); oof_dict['lgbm_balanced'] = oof

# ── 2. LightGBM avec scale_pos_weight ────────────────────────────────────────
ratio = (y == 0).sum() / (y == 1).sum()
def lgbm_scale():
    return lgb.LGBMClassifier(
        n_estimators=1200, learning_rate=0.02, max_depth=7,
        num_leaves=127, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=ratio, random_state=42, verbose=-1, n_jobs=-1)

r, oof = train_oof(lgbm_scale, f"LightGBM scale_pos_weight={ratio:.1f}")
results.append(r); oof_dict['lgbm_scale'] = oof

# ── 3. LightGBM + SMOTE ──────────────────────────────────────────────────────
def lgbm_plain():
    return lgb.LGBMClassifier(
        n_estimators=1200, learning_rate=0.02, max_depth=7,
        num_leaves=127, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1, n_jobs=-1)

r, oof = train_oof(lgbm_plain, "LightGBM + SMOTE", use_smote=True, smote_strategy=SMOTE)
results.append(r); oof_dict['lgbm_smote'] = oof

# ── 4. LightGBM + SMOTE-Tomek ────────────────────────────────────────────────
r, oof = train_oof(lgbm_plain, "LightGBM + SMOTETomek", use_smote=True, smote_strategy=SMOTETomek)
results.append(r); oof_dict['lgbm_smotetomek'] = oof

# ── 5. XGBoost balanced ──────────────────────────────────────────────────────
def xgb_balanced():
    return xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.02, max_depth=7,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=ratio, eval_metric='logloss',
        random_state=42, verbosity=0, n_jobs=-1)

r, oof = train_oof(xgb_balanced, "XGBoost balanced")
results.append(r); oof_dict['xgb_balanced'] = oof

# ── 6. XGBoost + SMOTE ───────────────────────────────────────────────────────
def xgb_plain():
    return xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.02, max_depth=7,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', random_state=42, verbosity=0, n_jobs=-1)

r, oof = train_oof(xgb_plain, "XGBoost + SMOTE", use_smote=True, smote_strategy=SMOTE)
results.append(r); oof_dict['xgb_smote'] = oof

# ── 7. Random Forest balanced ────────────────────────────────────────────────
def rf_balanced():
    return RandomForestClassifier(
        n_estimators=500, max_depth=15, min_samples_leaf=5,
        class_weight='balanced_subsample', random_state=42, n_jobs=-1)

r, oof = train_oof(rf_balanced, "Random Forest balanced")
results.append(r); oof_dict['rf_balanced'] = oof

# ── 8. Gradient Boosting (sklearn) ───────────────────────────────────────────
def gb_balanced():
    return GradientBoostingClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=5,
        subsample=0.8, random_state=42)

r, oof = train_oof(gb_balanced, "GradientBoosting sklearn")
results.append(r); oof_dict['gb_sklearn'] = oof

# ── 9. Logistic Regression baseline ──────────────────────────────────────────
def logreg():
    return LogisticRegression(
        class_weight='balanced', max_iter=1000, C=1.0, random_state=42)

r, oof = train_oof(logreg, "Logistic Regression balanced")
results.append(r); oof_dict['logreg'] = oof

# ── 10. LightGBM tuned (plus d'arbres, focal loss approx) ────────────────────
def lgbm_tuned():
    return lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.01, max_depth=8,
        num_leaves=200, subsample=0.75, colsample_bytree=0.7,
        min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
        class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1)

r, oof = train_oof(lgbm_tuned, "LightGBM tuned (2000 trees)")
results.append(r); oof_dict['lgbm_tuned'] = oof

# ── 11. LightGBM per-airport (meilleur approche precedente) ──────────────────
print("\n--- Modeles par aeroport ---")
oof_airport = np.zeros(len(y))
airport_results = {}

for airport in df['airport'].unique():
    mask = df['airport'] == airport
    X_ap = X.loc[mask]
    y_ap = y.loc[mask]
    g_ap = groups.loc[mask]

    gkf_ap = GroupKFold(n_splits=5)
    oof_ap = np.zeros(mask.sum())

    for tr, va in gkf_ap.split(X_ap, y_ap, g_ap):
        model = lgb.LGBMClassifier(
            n_estimators=1500, learning_rate=0.015, max_depth=7,
            num_leaves=127, subsample=0.8, colsample_bytree=0.8,
            class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1)
        model.fit(X_ap.iloc[tr], y_ap.iloc[tr])
        oof_ap[va] = model.predict_proba(X_ap.iloc[va])[:, 1]

    oof_airport[mask] = oof_ap

    # Best F1 threshold per airport
    prec_c, rec_c, thr_c = precision_recall_curve(y_ap, oof_ap)
    f1s = 2 * prec_c * rec_c / (prec_c + rec_c + 1e-8)
    best_idx = np.argmax(f1s)
    best_thr = thr_c[min(best_idx, len(thr_c)-1)]

    y_pred_ap = (oof_ap >= best_thr).astype(int)
    airport_results[airport] = {
        'auc': round(roc_auc_score(y_ap, oof_ap), 4),
        'f1': round(f1_score(y_ap, y_pred_ap), 4),
        'precision': round(precision_score(y_ap, y_pred_ap), 4),
        'recall': round(recall_score(y_ap, y_pred_ap), 4),
        'best_thr': round(best_thr, 3),
        'n_samples': int(mask.sum()),
        'n_pos': int(y_ap.sum()),
    }
    print(f"  {airport:12s} | AUC={airport_results[airport]['auc']:.4f} | "
          f"F1={airport_results[airport]['f1']:.4f} | thr={best_thr:.3f} | "
          f"n={mask.sum()}, pos={y_ap.sum()}")

r_ap = evaluate_oof(y, oof_airport, "LightGBM per-airport tuned")
results.append(r_ap)
oof_dict['lgbm_per_airport'] = oof_airport

# ── 12. Ensemble (moyenne des meilleurs) ──────────────────────────────────────
print("\n--- Ensemble ---")
top_models = ['lgbm_tuned', 'xgb_balanced', 'lgbm_per_airport']
oof_ensemble = np.mean([oof_dict[k] for k in top_models], axis=0)
r_ens = evaluate_oof(y, oof_ensemble, f"Ensemble ({'+'.join(top_models[:2])}+per_ap)")
results.append(r_ens)
oof_dict['ensemble'] = oof_ensemble

# ── 13. Stacking simple ──────────────────────────────────────────────────────
print("\n--- Stacking ---")
stack_feats = np.column_stack([oof_dict[k] for k in ['lgbm_balanced', 'lgbm_tuned', 'xgb_balanced', 'lgbm_per_airport', 'rf_balanced']])
from sklearn.linear_model import LogisticRegressionCV
oof_stack = np.zeros(len(y))
for tr, va in folds:
    lr = LogisticRegressionCV(cv=3, random_state=42, max_iter=1000)
    lr.fit(stack_feats[tr], y.iloc[tr])
    oof_stack[va] = lr.predict_proba(stack_feats[va])[:, 1]

r_stack = evaluate_oof(y, oof_stack, "Stacking (LR sur 5 modeles)")
results.append(r_stack)
oof_dict['stacking'] = oof_stack


# ══════════════════════════════════════════════════════════════════════════════
# TABLEAU RECAPITULATIF
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TABLEAU RECAPITULATIF (trie par F1)")
print("=" * 70)

df_results = pd.DataFrame(results).sort_values('f1', ascending=False)
print(df_results[['name', 'auc_roc', 'auc_pr', 'f1', 'precision', 'recall', 'mcc', 'best_threshold']].to_string(index=False))

# Sauvegarder les resultats
df_results.to_csv('model_comparison_results.csv', index=False)
print(f"\nResultats sauvegardes dans model_comparison_results.csv")

# Sauvegarder les resultats par aeroport
with open('airport_results.json', 'w') as f:
    json.dump(airport_results, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES DE COMPARAISON
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GENERATION DES FIGURES")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Comparaison des modeles - Data Battle 2026', fontsize=16, fontweight='bold')

# 1. Barplot F1 par modele
ax = axes[0, 0]
df_plot = df_results.sort_values('f1')
colors = ['#2ca02c' if 'per-airport' in n.lower() or 'ensemble' in n.lower() or 'stacking' in n.lower()
          else '#1f77b4' for n in df_plot['name']]
ax.barh(range(len(df_plot)), df_plot['f1'], color=colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(df_plot)))
ax.set_yticklabels(df_plot['name'], fontsize=8)
ax.set_xlabel('F1 Score')
ax.set_title('F1 Score par modele')
ax.axvline(x=0.433, color='red', linestyle='--', alpha=0.7, label='Baseline V3 (0.433)')
ax.legend(fontsize=8)
for i, v in enumerate(df_plot['f1']):
    ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=7)

# 2. AUC-ROC vs AUC-PR scatter
ax = axes[0, 1]
for _, row in df_results.iterrows():
    ax.scatter(row['auc_roc'], row['auc_pr'], s=100, zorder=5)
    ax.annotate(row['name'].split('(')[0].strip()[:20], (row['auc_roc'], row['auc_pr']),
                fontsize=6, ha='center', va='bottom')
ax.set_xlabel('AUC-ROC')
ax.set_ylabel('AUC-PR (plus informatif avec desequilibre)')
ax.set_title('AUC-ROC vs AUC-PR')
ax.grid(True, alpha=0.3)

# 3. Precision-Recall trade-off pour top 4
ax = axes[1, 0]
top4 = df_results.nlargest(4, 'f1')['name'].tolist()
for name in top4:
    key = [k for k, v in zip(oof_dict.keys(), [evaluate_oof.__code__] * len(oof_dict)) if True][0]
    # Find matching key
    for k in oof_dict:
        if k.replace('_', ' ') in name.lower().replace('(', '').replace(')', '') or name.lower().startswith(k.split('_')[0]):
            prec_c, rec_c, _ = precision_recall_curve(y, oof_dict[k])
            ax.plot(rec_c, prec_c, label=name[:30])
            break
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall (top 4 modeles)')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# 4. Resultats par aeroport
ax = axes[1, 1]
ap_names = list(airport_results.keys())
ap_f1 = [airport_results[a]['f1'] for a in ap_names]
ap_auc = [airport_results[a]['auc'] for a in ap_names]
x_pos = np.arange(len(ap_names))
w = 0.35
ax.bar(x_pos - w/2, ap_auc, w, label='AUC', color='#1f77b4', edgecolor='black', linewidth=0.5)
ax.bar(x_pos + w/2, ap_f1, w, label='F1', color='#ff7f0e', edgecolor='black', linewidth=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(ap_names, rotation=30, ha='right')
ax.set_ylabel('Score')
ax.set_title('Performances par aeroport (LightGBM per-airport)')
ax.legend()
for i in range(len(ap_names)):
    ax.text(x_pos[i] - w/2, ap_auc[i] + 0.01, f'{ap_auc[i]:.3f}', ha='center', fontsize=7)
    ax.text(x_pos[i] + w/2, ap_f1[i] + 0.01, f'{ap_f1[i]:.3f}', ha='center', fontsize=7)

plt.tight_layout()
plt.savefig('plots/14_model_comparison.png', dpi=150, bbox_inches='tight')
print("Figure sauvegardee : plots/14_model_comparison.png")


# ── Precision-Recall curves detaillees ────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 7))
for k in ['lgbm_balanced', 'lgbm_tuned', 'xgb_balanced', 'lgbm_per_airport', 'rf_balanced', 'ensemble', 'stacking', 'logreg']:
    if k in oof_dict:
        prec_c, rec_c, _ = precision_recall_curve(y, oof_dict[k])
        ap = average_precision_score(y, oof_dict[k])
        label_clean = k.replace('_', ' ').title()
        ax2.plot(rec_c, prec_c, label=f'{label_clean} (AP={ap:.3f})', linewidth=1.5)

ax2.axhline(y=y.mean(), color='gray', linestyle='--', alpha=0.5, label=f'Baseline aleatoire ({y.mean():.3f})')
ax2.set_xlabel('Recall', fontsize=12)
ax2.set_ylabel('Precision', fontsize=12)
ax2.set_title('Courbes Precision-Recall - Tous les modeles', fontsize=14)
ax2.legend(fontsize=8, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])
plt.tight_layout()
plt.savefig('plots/15_precision_recall_curves.png', dpi=150, bbox_inches='tight')
print("Figure sauvegardee : plots/15_precision_recall_curves.png")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRAINER LE MEILLEUR MODELE FINAL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ENTRAINEMENT DU MODELE FINAL")
print("=" * 70)

# Identifier le meilleur modele
best_row = df_results.iloc[0]  # Deja trie par F1
print(f"Meilleur F1 : {best_row['name']} avec F1={best_row['f1']}")
print()

# Entrainer les modeles finaux par aeroport (meilleure approche)
final_models = {}
final_features = FEAT
le_final = LabelEncoder()
le_final.fit(df['airport'])

for airport in df['airport'].unique():
    mask = df['airport'] == airport
    X_ap = X.loc[mask]
    y_ap = y.loc[mask]

    model = lgb.LGBMClassifier(
        n_estimators=1500, learning_rate=0.015, max_depth=7,
        num_leaves=127, subsample=0.8, colsample_bytree=0.8,
        class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1)
    model.fit(X_ap, y_ap)

    final_models[airport] = {
        'model': model,
        'best_thr': airport_results[airport]['best_thr'],
        'auc': airport_results[airport]['auc'],
        'f1': airport_results[airport]['f1'],
    }
    print(f"  {airport}: entraine sur {mask.sum()} exemples")

# Sauvegarder
model_pack = {
    'airport_models': final_models,
    'features': FEAT,
    'le': le_final,
    'comparison_results': results,
    'airport_details': airport_results,
}
with open('models/lgbm_per_airport.pkl', 'wb') as f:
    pickle.dump(model_pack, f)
print(f"\nModele final sauvegarde : models/lgbm_per_airport.pkl")
print(f"Features : {len(FEAT)}")

# Aussi entrainer un modele global LightGBM tuned (fallback)
model_global = lgb.LGBMClassifier(
    n_estimators=2000, learning_rate=0.01, max_depth=8,
    num_leaves=200, subsample=0.75, colsample_bytree=0.7,
    min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
    class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1)
model_global.fit(X, y)

global_pack = {
    'model': model_global,
    'features': FEAT,
    'le': le_final,
    'best_thr': df_results[df_results['name'].str.contains('tuned')].iloc[0]['best_threshold'],
}
with open('models/lgbm_v3.pkl', 'wb') as f:
    pickle.dump(global_pack, f)
print(f"Modele global sauvegarde : models/lgbm_v3.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE DETAILLEE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE (modele global)")
print("=" * 70)

fi = pd.DataFrame({
    'feature': FEAT,
    'importance': model_global.feature_importances_
}).sort_values('importance', ascending=False)

print(fi.head(20).to_string(index=False))

fig3, ax3 = plt.subplots(figsize=(10, 8))
top20 = fi.head(20).sort_values('importance')
ax3.barh(range(len(top20)), top20['importance'], color='#1f77b4', edgecolor='black', linewidth=0.5)
ax3.set_yticks(range(len(top20)))
ax3.set_yticklabels(top20['feature'])
ax3.set_xlabel('Importance (gain)')
ax3.set_title('Top 20 features les plus importantes (LightGBM global)')
plt.tight_layout()
plt.savefig('plots/16_feature_importance_detailed.png', dpi=150, bbox_inches='tight')
print("\nFigure sauvegardee : plots/16_feature_importance_detailed.png")


print("\n" + "=" * 70)
print("TERMINE")
print("=" * 70)
print(f"Resultats dans model_comparison_results.csv")
print(f"3 figures generees dans plots/")
print(f"Modeles sauvegardes dans models/")
