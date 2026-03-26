"""
Construit le notebook 02_model_prediction.ipynb avec les images embedees
directement dans les outputs (base64) pour qu'il soit auto-suffisant.
"""
import json
import base64
import os

def make_nb():
    cells = []

    def md(src):
        lines = src.split("\n")
        source = [l + "\n" for l in lines[:-1]] + [lines[-1]] if len(lines) > 1 else [src]
        cells.append({"cell_type": "markdown", "metadata": {}, "source": source})

    def code(src, outputs=None):
        lines = src.split("\n")
        source = [l + "\n" for l in lines[:-1]] + [lines[-1]] if len(lines) > 1 else [src]
        c = {"cell_type": "code", "metadata": {}, "source": source,
             "execution_count": None, "outputs": outputs or []}
        cells.append(c)

    def img_output(filepath, caption=""):
        """Cree un output display_data avec image PNG en base64."""
        if not os.path.exists(filepath):
            return {"output_type": "stream", "name": "stdout",
                    "text": [f"[Image manquante: {filepath}]\n"]}
        with open(filepath, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return {
            "output_type": "display_data",
            "metadata": {},
            "data": {
                "image/png": b64,
                "text/plain": [caption or filepath]
            }
        }

    def text_output(text):
        return {"output_type": "stream", "name": "stdout",
                "text": text.split("\n") if isinstance(text, str) else text}

    # ======================================================================
    # HEADER
    # ======================================================================
    md("""# Data Battle 2026 - Meteorage
## Prediction de la fin d'un orage

**Objectif** : Estimer pour chaque eclair nuage-sol (CG) la probabilite qu'il soit le dernier de l'alerte, afin de lever l'alerte orageuse plus tot tout en maintenant un risque < 2%.

| Metrique | LightGBM global | XGBoost global | Per-airport | Ensemble |
|---|---|---|---|---|
| AUC-ROC (OOF) | 0.9338 | **0.9365** | 0.9260 | 0.9354 |
| AUC-PR (OOF) | 0.4091 | **0.4190** | 0.3831 | 0.4161 |
| F1 optimal | 0.4398 | 0.4478 | 0.4315 | **0.4489** |
| Gain (eval) | 278.5 h | **548.3 h** | 573.8 h | 404.0 h |
| Risque (eval) | 1.90% | **1.76%** | 1.88% | 1.79% |

**5 aeroports** : Ajaccio, Bastia, Biarritz, Nantes, Pise | **Train** : 2016-2022 (769 alertes) | **Eval** : 2023-2025 (1081 alertes)""")

    # ======================================================================
    # 1. IMPORTS
    # ======================================================================
    md("## 1. Imports et chargement des donnees")
    code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              precision_recall_curve, f1_score,
                              precision_score, recall_score, matthews_corrcoef)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle, os, warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['figure.dpi'] = 120
sns.set_style('whitegrid')

TRAIN_PATH = 'segment_alerts_all_airports_train/segment_alerts_all_airports_train.csv'
EVAL_PATH  = 'segment_alerts_all_airports_eval.csv'

df_train = pd.read_csv(TRAIN_PATH)
df_train['date'] = pd.to_datetime(df_train['date'], utc=True)
df_eval  = pd.read_csv(EVAL_PATH)
df_eval['date'] = pd.to_datetime(df_eval['date'], utc=True)

for df in [df_train, df_eval]:
    df['icloud'] = df['icloud'].astype(str).str.strip().str.lower().map({'true':True,'false':False}).fillna(False)

print(f"Train : {df_train.shape}  |  Eval : {df_eval.shape}")
print(f"Train : {df_train['date'].min().date()} -> {df_train['date'].max().date()}")
print(f"Eval  : {df_eval['date'].min().date()} -> {df_eval['date'].max().date()}")
print(f"Aeroports : {sorted(df_train['airport'].unique())}")""")

    # ======================================================================
    # 2. EDA
    # ======================================================================
    md("## 2. Analyse exploratoire des donnees (EDA)")

    md("""### 2.1 Vue d'ensemble des donnees

Les donnees couvrent 5 aeroports europeens. Chaque eclair est caracterise par sa position (dist, azimuth), son intensite (amplitude, maxis), son type (CG ou IC) et son appartenance eventuelle a une alerte.""")

    code("""# Statistiques generales
labeled = df_train[df_train['airport_alert_id'].notnull()]
cg_labeled = labeled[~labeled['icloud']]

print("=== TRAIN (2016-2022) ===")
print(f"Eclairs totaux         : {len(df_train):,}")
print(f"  dont CG              : {(~df_train['icloud']).sum():,}")
print(f"  dont IC              : {df_train['icloud'].sum():,}")
print(f"  ratio IC/CG          : {df_train['icloud'].sum() / (~df_train['icloud']).sum():.2f}")
print(f"Eclairs en alerte (CG) : {len(cg_labeled):,}")
print(f"Alertes labellisees    : {cg_labeled['airport_alert_id'].nunique()}")
print()

target = cg_labeled['is_last_lightning_cloud_ground']
n_pos = (target.astype(str).str.lower() == 'true').sum()
n_neg = len(target) - n_pos
print(f"Classe positive (dernier eclair)    : {n_pos:,} ({n_pos/len(target)*100:.2f}%)")
print(f"Classe negative (eclair non-dernier): {n_neg:,} ({n_neg/len(target)*100:.2f}%)")
print(f"Ratio negatif/positif : {n_neg/n_pos:.0f}:1")
print()

print("=== REPARTITION PAR AEROPORT ===")
for ap in sorted(df_train['airport'].unique()):
    sub = df_train[df_train['airport'] == ap]
    n_cg = (~sub['icloud']).sum()
    n_ic = sub['icloud'].sum()
    n_al = sub[sub['airport_alert_id'].notnull()]['airport_alert_id'].nunique()
    print(f"  {ap:<12} CG={n_cg:>6,}  IC={n_ic:>7,}  Alertes={n_al:>3}  IC/CG={n_ic/n_cg:.2f}")""")

    md("### 2.2 Caracteristiques par aeroport")
    code("# Voir figure ci-dessous", outputs=[
        img_output("plots/01_airport_analysis.png",
                   "Figure 1 - Distribution des eclairs CG/IC, alertes et distances par aeroport")])

    md("### 2.3 Saisonnalite et patterns temporels\n\nL'activite orageuse est concentree en ete (juin-septembre, 68% des eclairs) avec un pic horaire entre 14h et 18h UTC, coherent avec la convection thermique diurne.")
    code("# Voir figure ci-dessous", outputs=[
        img_output("plots/02_temporal_analysis.png",
                   "Figure 2 - Saisonnalite mensuelle et repartition horaire par aeroport")])

    md("### 2.4 Structure des alertes\n\nLa duree mediane d'une alerte est de 18 minutes (moyenne : 30 min). Le nombre median d'eclairs CG par alerte est de 15.")
    code("# Voir figure ci-dessous", outputs=[
        img_output("plots/03_alert_analysis.png",
                   "Figure 3 - Distribution de duree, nombre d'eclairs et evolution temporelle des alertes")])

    md("### 2.5 Distribution spatiale\n\nL'azimuth (direction de l'eclair) revele des trajectoires preferentielles propres a chaque aeroport. Cette variable est la plus importante du modele.")
    code("# Voir figure ci-dessous", outputs=[
        img_output("plots/04_spatial_analysis.png",
                   "Figure 4 - Distribution azimutale et carte de densite des impacts")])

    md("""### 2.6 Signaux discriminants du dernier eclair CG

La comparaison statistique entre le dernier eclair et les eclairs non-terminaux revele des signaux physiquement interpretables :

| Caracteristique | Eclairs normaux | Dernier eclair | Ecart |
|---|---|---|---|
| Temps inter-eclairs moyen | ~45 s | ~120 s | +167% |
| Distance a l'aeroport | 20.2 km | 22.8 km | +13% |
| Amplitude absolue moyenne | 12.4 kA | 10.1 kA | -19% |
| Ratio IC/(IC+CG) sur 5 min | 0.78 | 0.85 | +9% |
| Densite CG (10 dernieres min) | 4.2/min | 0.8/min | -81% |""")

    code("# Voir figure ci-dessous", outputs=[
        img_output("plots/05_last_lightning_analysis.png",
                   "Figure 5 - Comparaison dernier eclair CG vs eclairs normaux")])

    code("# Exemples d'alertes annotees", outputs=[
        img_output("plots/06_alert_examples.png",
                   "Figure 6 - Exemples d'alertes avec evolution temporelle")])

    md("### 2.7 Matrice de correlation\n\nLes correlations lineaires avec la cible sont quasi nulles, ce qui est attendu avec un desequilibre de 4.64% et des relations non-lineaires entre features. C'est pourquoi les modeles a base d'arbres (LightGBM, XGBoost) sont bien adaptes : ils capturent les interactions non-lineaires entre features.")
    code("# Voir figure ci-dessous", outputs=[
        img_output("plots/06_correlation_matrix.png",
                   "Figure 7 - Matrice de correlation des features principales")])

    # ======================================================================
    # 3. FEATURE ENGINEERING
    # ======================================================================
    md("""## 3. Feature Engineering

### Contrainte fondamentale : causalite stricte

Chaque feature ne peut utiliser que les informations disponibles au moment t.
Les variables suivantes ont ete exclues (data leakage) :
- `total_in_alert` : nombre total d'eclairs de l'alerte (inconnu en temps reel)
- `rel_pos = rank / total` : utilise le total futur
- `remaining = total - rank - 1` : vaut 0 pour le dernier eclair = predicteur parfait mais frauduleux

Leur suppression a fait chuter l'AUC de 1.000 a 0.920.

### Categories de features (62 au total)

| Categorie | Variables | Signal physique |
|---|---|---|
| Position dans l'alerte | rank_cg, t_since_start_s | Probabilite croissante de fin |
| Temps inter-eclairs | dt_prev_s, dt_mean_W, dt_max_W, dt_ema_W | Espacement croissant = fin |
| Distance et tendance | dist, dist_mean_W, dist_trend_W | Orage qui s'eloigne |
| Amplitude | amp_abs, maxis, amp_mean_W, amp_trend_W | Declin energetique |
| Densite CG | n_cg_2/5/10/15/30min, rate_decline_cg | Ralentissement activite |
| Contexte IC | n_ic_2/5/10/30min, ratio_ic_cg, ic_trend | Precurseur de fin |
| Metadata | month, hour, airport_enc | Saisonnalite et specificites locales |""")

    code("""def make_features(df_cg, df_full=None):
    \"\"\"Feature engineering causal complet (62 features).\"\"\"
    df = df_cg.sort_values(['airport', 'airport_alert_id', 'date']).reset_index(drop=True)
    grp = df.groupby(['airport', 'airport_alert_id'])

    # Position dans l'alerte
    df['rank_cg']         = grp.cumcount()
    df['t_since_start_s'] = (df['date'] - grp['date'].transform('min')).dt.total_seconds()
    df['dt_prev_s']       = grp['date'].diff().dt.total_seconds().fillna(0)
    df['amp_abs']         = df['amplitude'].abs()
    df['amp_sign']        = np.sign(df['amplitude'])

    # Rolling windows sur fenetres 3, 5, 10, 20
    for w in [3, 5, 10, 20]:
        df[f'dt_mean_{w}']    = grp['dt_prev_s'].transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f'dt_max_{w}']     = grp['dt_prev_s'].transform(lambda x: x.rolling(w, min_periods=1).max())
        df[f'dt_std_{w}']     = grp['dt_prev_s'].transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0))
        df[f'dt_ema_{w}']     = grp['dt_prev_s'].transform(lambda x: x.ewm(span=w, adjust=False).mean())
        df[f'dist_mean_{w}']  = grp['dist'].transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f'dist_trend_{w}'] = grp['dist'].transform(
            lambda x: x.rolling(w, min_periods=2).apply(
                lambda v: np.polyfit(range(len(v)), v, 1)[0] if len(v) > 1 else 0, raw=True))
        df[f'amp_mean_{w}']   = grp['amplitude'].transform(lambda x: x.abs().rolling(w, min_periods=1).mean())
        df[f'amp_trend_{w}']  = grp['amplitude'].transform(
            lambda x: x.abs().rolling(w, min_periods=2).apply(
                lambda v: np.polyfit(range(len(v)), v, 1)[0] if len(v) > 1 else 0, raw=True))

    # Densite temporelle CG (count in sliding window)
    for wm, wns in [(2, 120e9), (5, 300e9), (10, 600e9), (15, 900e9), (30, 1800e9)]:
        wns = np.timedelta64(int(wns), 'ns')
        col = np.zeros(len(df))
        for (ap, aid), idx in grp.groups.items():
            dates_v = df.loc[idx, 'date'].values
            for i in range(len(dates_v)):
                col[idx[i]] = np.sum(dates_v[:i+1] >= dates_v[i] - wns)
        df[f'n_cg_{wm}min'] = col

    df['rate_decline_cg'] = df['n_cg_5min'] / (df['n_cg_10min'] + 1)

    # Contexte IC (eclairs intra-nuage)
    for c in ['n_ic_2min','n_ic_5min','n_ic_10min','n_ic_30min',
              'n_all_2min','n_all_5min','n_all_10min']:
        df[c] = 0.0
    if df_full is not None:
        for airport in df['airport'].unique():
            mask = df['airport'] == airport
            ic_d  = np.sort(df_full[(df_full['airport']==airport) & df_full['icloud']]['date'].values)
            all_d = np.sort(df_full[df_full['airport']==airport]['date'].values)
            cg_d  = df.loc[mask, 'date'].values
            for wns, col_name in [(120e9,'n_ic_2min'), (300e9,'n_ic_5min'),
                                   (600e9,'n_ic_10min'), (1800e9,'n_ic_30min')]:
                wns = np.timedelta64(int(wns), 'ns')
                df.loc[mask, col_name] = [np.searchsorted(ic_d, t+np.timedelta64(1,'ns')) -
                                           np.searchsorted(ic_d, t - wns) for t in cg_d]
            for wns, col_name in [(120e9,'n_all_2min'), (300e9,'n_all_5min'), (600e9,'n_all_10min')]:
                wns = np.timedelta64(int(wns), 'ns')
                df.loc[mask, col_name] = [np.searchsorted(all_d, t+np.timedelta64(1,'ns')) -
                                           np.searchsorted(all_d, t - wns) for t in cg_d]

    df['ratio_ic_cg_5min']  = df['n_ic_5min']  / (df['n_cg_5min']  + 1)
    df['ratio_ic_cg_10min'] = df['n_ic_10min'] / (df['n_cg_10min'] + 1)
    df['ic_trend']          = df['n_ic_5min'] - df['n_ic_10min'] / 2
    df['total_activity']    = df['n_all_5min']
    df['activity_decline']  = df['n_all_5min'] / (df['n_all_10min'] + 1)
    df['month'] = df['date'].dt.month
    df['hour']  = df['date'].dt.hour
    return df

print("Feature engineering defini (62 features causales)")""")

    # ======================================================================
    # 4. COMPARAISON DES MODELES
    # ======================================================================
    md("""## 4. Comparaison des modeles

### Strategie de comparaison

Nous comparons systematiquement plusieurs familles de modeles avec differentes strategies de gestion du desequilibre de classes (4.64% positifs, ratio 1:20) :

**Modeles testes :**
- **LightGBM** : gradient boosting rapide, avec `class_weight='balanced'` et `scale_pos_weight`
- **XGBoost** : gradient boosting, avec `scale_pos_weight` proportionnel au ratio de desequilibre
- **Random Forest** : bagging, avec `class_weight='balanced_subsample'`
- **Logistic Regression** : baseline lineaire avec `class_weight='balanced'`
- **Ensemble** : moyenne des scores de 3 modeles (LightGBM tuned + XGBoost + per-airport)
- **Stacking** : meta-modele (Logistic Regression) sur les predictions de 5 modeles de base

**Validation** : GroupKFold(5) par alerte pour eviter la contamination temporelle.

**Metriques** : F1-score (sur seuil optimal), AUC-ROC, AUC-PR (Average Precision, plus informatif que l'AUC-ROC en cas de desequilibre), MCC (Matthews Correlation Coefficient).

### Pourquoi ces metriques ?

L'accuracy est inadaptee : un classifieur trivial qui predit toujours "non-dernier" obtient 95.4% d'accuracy. Le F1 penalise les faux positifs et faux negatifs de maniere equilibree. L'AUC-PR est sensible au desequilibre et mesure la qualite de la discrimination des positifs rares. Le MCC est la metrique la plus equilibree pour la classification binaire desequilibree.""")

    code("""# Configuration commune
FEAT = [c for c in X.columns]  # 62 features causales
gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(X, y, groups))

# Resultats de la comparaison (pre-calcules)
# Voir model_comparison.py pour le code complet

results_table = '''
| Modele                              | AUC-ROC | AUC-PR | F1     | Precision | Rappel | MCC   |
|-------------------------------------|---------|--------|--------|-----------|--------|-------|
| Ensemble (LightGBM+XGBoost+per-ap) | 0.9354  | 0.4161 | 0.4489 | 0.379     | 0.550  | 0.425 |
| XGBoost tuned (1500 trees)          | 0.9365  | 0.4190 | 0.4478 | 0.350     | 0.622  | 0.433 |
| Stacking (LR sur 5 modeles)         | 0.9352  | 0.4157 | 0.4469 | 0.365     | 0.576  | 0.426 |
| LightGBM tuned (2000 trees)         | 0.9338  | 0.4091 | 0.4398 | 0.355     | 0.577  | 0.419 |
| Random Forest balanced              | 0.9345  | 0.4049 | 0.4392 | 0.360     | 0.564  | 0.417 |
| XGBoost balanced (1000 trees)       | 0.9327  | 0.4053 | 0.4331 | 0.370     | 0.522  | 0.407 |
| LightGBM per-airport                | 0.9260  | 0.3831 | 0.4315 | 0.350     | 0.563  | 0.410 |
| LightGBM balanced                   | 0.9308  | 0.3975 | 0.4312 | 0.352     | 0.555  | 0.409 |
| LightGBM scale_pos_weight=21        | 0.9306  | 0.4012 | 0.4306 | 0.359     | 0.537  | 0.406 |
| Logistic Regression balanced        | 0.9170  | 0.2919 | 0.3888 | 0.298     | 0.558  | 0.369 |
'''
print(results_table)""")

    md("### Comparaison visuelle des modeles")
    code("# Comparaison des F1, AUC-ROC vs AUC-PR, courbes PR, performances par aeroport", outputs=[
        img_output("plots/14_model_comparison.png",
                   "Figure 8 - Comparaison des modeles : F1, AUC-ROC vs AUC-PR, courbes PR, par aeroport")])

    md("### Courbes Precision-Recall detaillees")
    code("# Courbes PR pour tous les modeles", outputs=[
        img_output("plots/15_precision_recall_curves.png",
                   "Figure 9 - Courbes Precision-Recall de tous les modeles testes")])

    md("### Courbes ROC")
    code("# Courbes ROC", outputs=[
        img_output("plots/16_roc_curves.png",
                   "Figure 10 - Courbes ROC de tous les modeles testes")])

    md("""### Analyse des resultats de classification

**Observations cles :**

1. **La Logistic Regression est nettement inferieure** (F1=0.389, AUC-PR=0.292) : les correlations lineaires avec la cible sont quasi nulles (visible sur la matrice de correlation). Les modeles a base d'arbres capturent les interactions non-lineaires essentielles.

2. **XGBoost et LightGBM sont proches** mais XGBoost tuned a le meilleur AUC-ROC (0.9365) et AUC-PR (0.4190). LightGBM tuned a un bon F1 (0.4398) avec un bon equilibre precision/rappel.

3. **Le Random Forest est competitif** (F1=0.4392) grace au bagging qui reduit la variance, mais il est plus lent a entrainer.

4. **L'ensemble et le stacking ameliorent le F1** (0.449) en combinant les forces de modeles complementaires. La diversite des modeles (boosting + per-airport) est cle.

5. **`class_weight='balanced'` vs `scale_pos_weight`** : performances quasi identiques. La reponderation est suffisante, pas besoin de SMOTE.

6. **Les modeles per-airport** ont un AUC global plus faible (0.926 vs 0.937) car chaque sous-modele a moins de donnees, mais ils capturent les specificites locales.""")

    # ======================================================================
    # 5. FEATURE IMPORTANCE
    # ======================================================================
    md("## 5. Importance des features")

    code("# Feature importance detaillee (LightGBM global tuned)", outputs=[
        img_output("plots/17_feature_importance_final.png",
                   "Figure 11 - Top 20 features par importance (gain moyen LightGBM global)")])

    md("### Feature importance precedente (modeles V2/V3)")
    code("# Feature importance des versions precedentes", outputs=[
        img_output("plots/07_feature_importance.png",
                   "Figure 12 - Importance des features (versions V2 et V3)")])

    md("""### Interpretation des top features

| Rang | Feature | Importance | Interpretation physique |
|---|---|---|---|
| 1 | azimuth | 7908 | Direction de l'orage : trajectoire de deplacement |
| 2 | maxis | 7494 | Courant crete : declin energetique de la cellule |
| 3 | dist | 5771 | Distance a l'aeroport : orage qui s'eloigne |
| 4 | dist_mean_3 | 5293 | Distance moyenne recente : tendance d'eloignement |
| 5 | amplitude | 5196 | Intensite du courant : energie residuelle |
| 6 | activity_decline | 4932 | Ratio activite 5min/10min : ralentissement |
| 7 | hour | 4625 | Heure UTC : convection diurne (pic 14h-18h) |
| 8 | amp_abs | 4505 | Amplitude absolue : intensite sans signe |
| 9 | dist_trend_3 | 4477 | Tendance de distance sur 3 eclairs : vitesse d'eloignement |
| 10 | n_ic_30min | 4324 | Nombre d'eclairs IC en 30min : contexte orageux global |

Les features spatiales (azimuth, dist, dist_mean) et d'intensite (maxis, amplitude) dominent. Le contexte IC sur 30 minutes est aussi tres informatif, confirmant l'interet d'integrer les eclairs intra-nuage.""")

    # ======================================================================
    # 6. EVALUATION OPERATIONNELLE
    # ======================================================================
    md("""## 6. Evaluation operationnelle (protocole Meteorage)

Le protocole officiel definit :
- **Gain** : temps recupere vs la regle fixe de 30 min (dernier eclair + 30 min)
- **Risque** : proportion d'eclairs < 3 km manques apres la levee d'alerte
- **Contrainte** : R < 2%
- **Theta** : seuil de confiance pour declarer la fin de l'alerte""")

    md("""### Resultats operationnels (eval 2023-2025, 1081 alertes)

| Modele | Theta | Gain (heures) | Risque | Gain par alerte |
|---|---|---|---|---|
| Baseline (30 min fixe) | - | 0 | 0.00% | 0 min |
| LightGBM per-airport | 0.60 | 573.8 | 1.88% | 31.8 min |
| LightGBM global tuned | 0.90 | 278.5 | 1.90% | 15.5 min |
| **XGBoost global tuned** | **0.85** | **548.3** | **1.76%** | **30.4 min** |
| Ensemble | 0.80 | 404.0 | 1.79% | 22.4 min |

Le XGBoost global avec theta=0.85 offre le meilleur compromis : 548.3 heures de gain (30 min par alerte en moyenne) avec un risque de 1.76%.""")

    code("# Courbes Gain et Risque vs theta", outputs=[
        img_output("plots/18_operational_evaluation.png",
                   "Figure 13 - Courbes Gain et Risque en fonction de theta pour les 4 modeles")])

    code("# Compromis Gain vs Risque", outputs=[
        img_output("plots/19_gain_vs_risk.png",
                   "Figure 14 - Gain vs Risque : visualisation du compromis operationnel")])

    md("### Resultats anciens (modeles V2/V3)")
    code("# Resultats operationnels des versions precedentes", outputs=[
        img_output("plots/10_evaluation_results.png",
                   "Figure 15 - Gain et risque par modele (versions V2/V3)")])

    code("# Courbe Gain-Risque V3", outputs=[
        img_output("plots/11_eval_gain_risk.png",
                   "Figure 16 - Courbe Gain-Risque vs theta (modele par aeroport V3)")])

    # ======================================================================
    # 7. RESULTATS PAR AEROPORT
    # ======================================================================
    md("## 7. Resultats par aeroport")

    md("""### Performances de classification par aeroport

| Aeroport | AUC | F1 | Seuil F1-optimal | Nombre d'alertes | Profil orageux |
|---|---|---|---|---|---|
| Ajaccio | 0.933 | 0.455 | 0.248 | 105 | Convection mediterraneenne intense et courte |
| Bastia | 0.923 | 0.395 | 0.282 | 234 | Orages mediterraneens, variabilite moderee |
| Biarritz | 0.924 | 0.467 | 0.132 | 188 | Orages atlantiques, trajectoires O-E variables |
| Nantes | 0.905 | 0.426 | 0.079 | 49 | Orages continentaux, peu d'alertes |
| Pise | 0.931 | 0.441 | 0.422 | 193 | Orages liguriens, fort ratio IC/CG |

La grande variabilite des seuils optimaux (0.079 pour Nantes vs 0.422 pour Pise) justifie la specialisation par aeroport : chaque lieu a une dynamique orageuse propre.""")

    code("# Resultats V3 detailles", outputs=[
        img_output("plots/12_model_v3_results.png",
                   "Figure 17 - Resultats V3 : courbe gain-risque et distribution par aeroport")])

    code("# Comparaison finale des approches", outputs=[
        img_output("plots/13_comparison_final.png",
                   "Figure 18 - Comparaison finale des approches V2 vs V3 vs par aeroport")])

    # ======================================================================
    # 8. CONCLUSIONS
    # ======================================================================
    md("""## 8. Conclusions et perspectives

### Contributions principales

1. **Comparaison multi-modeles rigoureuse** : 10 configurations testees (LightGBM, XGBoost, Random Forest, Logistic Regression, Ensemble, Stacking) avec validation GroupKFold(5) par alerte. Le XGBoost tuned domine en evaluation operationnelle (548.3h de gain, R=1.76%).

2. **Gestion du desequilibre** : `class_weight='balanced'` et `scale_pos_weight` sont equivalents et suffisants. Le SMOTE n'apporte pas d'amelioration sur ce probleme.

3. **Contexte IC determinant** : les eclairs intra-nuage (IC) constituent un signal precurseur de la fin d'orage. Leur integration ameliore l'AUC de +1.3% et le F1 de +10%.

4. **Specialisation par aeroport** : les modeles per-airport capturent les specificites meteorologiques locales et obtiennent 573.8h de gain (R=1.88%).

5. **Gain operationnel massif** : jusqu'a 30 minutes recuperees par alerte vs la regle fixe de 30 minutes, soit une reduction de moitie du temps d'immobilisation.

### Limites

- Precision moderee (~0.35-0.38) : certaines levees anticipees sont des faux positifs, mais le risque reste contenu sous 2%
- Modele eclair-par-eclair, sans memoire sequentielle explicite (les rolling windows compensent partiellement)
- Absence de donnees meteorologiques contextuelles (CAPE, vent, humidite, images satellite)

### Pistes d'amelioration

- Modeles sequentiels (LSTM, Transformer) sur la trajectoire temporelle de l'alerte
- Calibration des probabilites (Platt Scaling, Isotonic Regression) pour ameliorer le compromis gain-risque
- Post-processing : exiger K decisions consecutives avant levee pour reduire les faux positifs
- Integration de donnees NWP (previsions numeriques) et images satellite/radar""")

    # ── BUILD NOTEBOOK ──────────────────────────────────────────────────
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "cells": cells
    }
    with open("02_model_prediction.ipynb", "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Notebook genere : {len(cells)} cellules, images embeddees")

make_nb()
