"""
Data Battle 2026 - Meteorage
Application Streamlit : Prediction de fin d'orage
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
import os
from datetime import timedelta

st.set_page_config(
    page_title="OragEnd - Prediction de fin d'orage",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Chargement des modeles ────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)

@st.cache_resource
def load_models():
    models = {}
    for name, fname in [("per_airport", "lgbm_per_airport.pkl"),
                         ("v3", "lgbm_v3.pkl")]:
        p = os.path.join(BASE, "models", fname)
        if os.path.exists(p):
            with open(p, "rb") as f:
                models[name] = pickle.load(f)
    return models

@st.cache_data
def load_dataset(path, nrows=500_000):
    df = pd.read_csv(path, nrows=nrows)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["icloud"] = df["icloud"].map({"True": True, "False": False}).fillna(df["icloud"]).astype(bool)
    return df

# ── Feature engineering (identique au notebook) ──────────────────────────────
def make_features(df_cg, df_full=None):
    df = df_cg.sort_values(["airport", "airport_alert_id", "date"]).reset_index(drop=True)
    grp = df.groupby(["airport", "airport_alert_id"])
    df["rank_cg"]         = grp.cumcount()
    df["t_since_start_s"] = (df["date"] - grp["date"].transform("min")).dt.total_seconds()
    df["dt_prev_s"]       = grp["date"].diff().dt.total_seconds().fillna(0)
    df["amp_abs"]         = df["amplitude"].abs()
    df["amp_sign"]        = np.sign(df["amplitude"])

    for w in [3, 5, 10, 20]:
        df[f"dt_mean_{w}"]    = grp["dt_prev_s"].transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f"dt_max_{w}"]     = grp["dt_prev_s"].transform(lambda x: x.rolling(w, min_periods=1).max())
        df[f"dt_std_{w}"]     = grp["dt_prev_s"].transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0))
        df[f"dt_ema_{w}"]     = grp["dt_prev_s"].transform(lambda x: x.ewm(span=w, adjust=False).mean())
        df[f"dist_mean_{w}"]  = grp["dist"].transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f"dist_trend_{w}"] = grp["dist"].transform(
            lambda x: x.rolling(w, min_periods=2).apply(
                lambda v: np.polyfit(range(len(v)), v, 1)[0] if len(v) > 1 else 0, raw=True))
        df[f"amp_mean_{w}"]   = grp["amplitude"].transform(lambda x: x.abs().rolling(w, min_periods=1).mean())
        df[f"amp_trend_{w}"]  = grp["amplitude"].transform(
            lambda x: x.abs().rolling(w, min_periods=2).apply(
                lambda v: np.polyfit(range(len(v)), v, 1)[0] if len(v) > 1 else 0, raw=True))

    def rtc(dates_v, wns):
        r = np.zeros(len(dates_v))
        for i in range(len(dates_v)):
            r[i] = np.sum(dates_v[:i+1] >= dates_v[i] - wns)
        return r

    for wm, wns in [(2, np.timedelta64(120*10**9,"ns")), (5, np.timedelta64(300*10**9,"ns")),
                    (10, np.timedelta64(600*10**9,"ns")), (15, np.timedelta64(900*10**9,"ns")),
                    (30, np.timedelta64(1800*10**9,"ns"))]:
        col = np.zeros(len(df))
        for (ap, aid), idx in grp.groups.items():
            col[idx] = rtc(df.loc[idx, "date"].values, wns)
        df[f"n_cg_{wm}min"] = col

    df["rate_decline_cg"] = df["n_cg_5min"] / (df["n_cg_10min"] + 1)

    ic_cols = ["n_ic_2min","n_ic_5min","n_ic_10min","n_ic_30min",
               "n_all_2min","n_all_5min","n_all_10min"]
    for c in ic_cols:
        df[c] = 0.0
    if df_full is not None:
        for airport in df["airport"].unique():
            mask = df["airport"] == airport
            ic_d  = np.sort(df_full[(df_full["airport"]==airport) & df_full["icloud"]]["date"].values)
            all_d = np.sort(df_full[df_full["airport"]==airport]["date"].values)
            cg_d  = df.loc[mask, "date"].values
            for wns, col in [(np.timedelta64(120*10**9,"ns"),"n_ic_2min"),
                              (np.timedelta64(300*10**9,"ns"),"n_ic_5min"),
                              (np.timedelta64(600*10**9,"ns"),"n_ic_10min"),
                              (np.timedelta64(1800*10**9,"ns"),"n_ic_30min")]:
                df.loc[mask, col] = [np.searchsorted(ic_d, t+np.timedelta64(1,"ns")) -
                                      np.searchsorted(ic_d, t - wns) for t in cg_d]
            for wns, col in [(np.timedelta64(120*10**9,"ns"),"n_all_2min"),
                              (np.timedelta64(300*10**9,"ns"),"n_all_5min"),
                              (np.timedelta64(600*10**9,"ns"),"n_all_10min")]:
                df.loc[mask, col] = [np.searchsorted(all_d, t+np.timedelta64(1,"ns")) -
                                      np.searchsorted(all_d, t - wns) for t in cg_d]

    df["ratio_ic_cg_5min"]  = df["n_ic_5min"]  / (df["n_cg_5min"]  + 1)
    df["ratio_ic_cg_10min"] = df["n_ic_10min"] / (df["n_cg_10min"] + 1)
    df["ic_trend"]          = df["n_ic_5min"] - df["n_ic_10min"] / 2
    df["total_activity"]    = df["n_all_5min"]
    df["activity_decline"]  = df["n_all_5min"] / (df["n_all_10min"] + 1)
    df["month"]             = df["date"].dt.month
    df["hour"]              = df["date"].dt.hour
    return df


def predict(df_feat, models, airport):
    FEAT_AP = models["per_airport"]["features"] if "per_airport" in models else []
    for m in FEAT_AP:
        if m not in df_feat.columns:
            df_feat[m] = 0.0
    if "per_airport" in models and airport in models["per_airport"]["airport_models"]:
        mdl = models["per_airport"]["airport_models"][airport]["model"]
        return mdl.predict_proba(df_feat[FEAT_AP].astype(float))[:, 1]
    elif "v3" in models:
        feats_v3 = models["v3"]["features"]
        for m in feats_v3:
            if m not in df_feat.columns:
                df_feat[m] = 0.0
        le = models["v3"]["le"]
        df_feat["airport_enc"] = le.transform([airport])[0] if airport in le.classes_ else 0
        return models["v3"]["model"].predict_proba(df_feat[feats_v3].astype(float))[:, 1]
    return np.zeros(len(df_feat))


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
models = load_models()

with st.sidebar:
    st.title("OragEnd")
    st.caption("Prediction de fin d'orage - Data Battle 2026")
    st.divider()

    page = st.radio("Navigation", [
        "Prediction en temps reel",
        "Analyse exploratoire",
        "Guide d'utilisation",
    ])
    st.divider()

    theta = st.slider("Seuil de decision (theta)", 0.50, 0.99, 0.90, 0.01,
                       help="Premier eclair avec score >= theta declenchera la levee d'alerte. "
                            "Plus theta est eleve, moins de fausses alarmes mais gain reduit.")

    st.divider()
    st.markdown("**Performances (eval 2023-2025)**")
    st.markdown(f"- Gain : **213.8 h** sur 612 alertes")
    st.markdown(f"- Risque : **1.15%** (< 2%)")
    st.markdown(f"- ~21 min recuperees par alerte")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 : PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
if page == "Prediction en temps reel":

    st.header("Prediction de fin d'orage")
    st.markdown("Selectionnez une alerte du dataset pour visualiser la prediction du modele en temps reel.")

    # Choix du dataset
    eval_path = os.path.join(BASE, "segment_alerts_all_airports_eval.csv")
    train_path = os.path.join(BASE, "segment_alerts_all_airports_train",
                               "segment_alerts_all_airports_train.csv")
    datasets = {}
    if os.path.exists(eval_path):
        datasets["Evaluation (2023-2025)"] = eval_path
    if os.path.exists(train_path):
        datasets["Entrainement (2016-2022)"] = train_path

    if not datasets:
        st.error("Aucun fichier de donnees trouve.")
        st.stop()

    col_ds, col_ap, col_al = st.columns([2, 1, 1])
    with col_ds:
        ds_name = st.selectbox("Dataset", list(datasets.keys()))
    df_all = load_dataset(datasets[ds_name])
    labeled = df_all[df_all["airport_alert_id"].notnull()]

    with col_ap:
        airports_avail = sorted(labeled["airport"].unique())
        airport_sel = st.selectbox("Aeroport", airports_avail)

    with col_al:
        alerts_avail = sorted(labeled[labeled["airport"] == airport_sel]["airport_alert_id"].unique())
        alert_sel = st.selectbox("Alerte", alerts_avail,
                                  format_func=lambda x: f"#{int(x)} ({labeled[(labeled['airport']==airport_sel)&(labeled['airport_alert_id']==x)&(~labeled['icloud'])].shape[0]} eclairs)")

    # Extraire l'alerte
    df_alert_full = labeled[(labeled["airport"] == airport_sel) &
                             (labeled["airport_alert_id"] == alert_sel)]
    df_cg = df_alert_full[~df_alert_full["icloud"]].copy()

    if len(df_cg) == 0:
        st.warning("Aucun eclair CG dans cette alerte.")
        st.stop()

    # Feature engineering + prediction
    with st.spinner("Calcul des features et prediction..."):
        df_feat = make_features(df_cg.copy(), df_all)
        scores = predict(df_feat, models, airport_sel)
        df_feat["confidence"] = scores

    sub = df_feat.sort_values("date").reset_index(drop=True)
    t0 = sub["date"].min()
    sub["t_min"] = (sub["date"] - t0).dt.total_seconds() / 60

    last_date     = sub["date"].max()
    baseline_end  = last_date + pd.Timedelta(minutes=30)
    pred_above    = sub[sub["confidence"] >= theta]
    predicted_end = pred_above["date"].min() if len(pred_above) > 0 else None
    gain_min      = (baseline_end - predicted_end).total_seconds() / 60 if predicted_end else 0
    dur_min       = (last_date - t0).total_seconds() / 60

    # ── KPIs ──────────────────────────────────────────────────────────────────
    st.divider()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Eclairs CG", len(sub))
    c2.metric("Duree", f"{dur_min:.0f} min")
    c3.metric("Score max", f"{sub['confidence'].max():.1%}")
    if predicted_end:
        c4.metric("Gain estime", f"{gain_min:.0f} min", delta=f"+{gain_min:.0f} min vs 30 min fixe")
    else:
        c4.metric("Gain estime", "0", delta="Seuil non atteint")
    c5.metric(f"Eclairs >= theta ({theta:.0%})", len(pred_above))

    # ── Graphique principal (3 sous-graphiques) ───────────────────────────────
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.45, 0.30, 0.25],
        subplot_titles=[
            "Score de confiance (le modele estime si l'eclair est le dernier)",
            "Distance a l'aeroport (km)",
            "Activite CG (eclairs dans les 5 dernieres min)"
        ],
        vertical_spacing=0.08,
    )

    # Score de confiance
    fig.add_trace(go.Scatter(
        x=sub["t_min"], y=sub["confidence"],
        mode="lines+markers", name="Score",
        line=dict(color="#1f77b4", width=2.5),
        marker=dict(size=5),
        hovertemplate="t=%{x:.1f} min<br>Score=%{y:.3f}<extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=theta, line_dash="dash", line_color="red",
                  annotation_text=f"theta = {theta:.2f}", row=1, col=1)
    if predicted_end:
        pred_t = (predicted_end - t0).total_seconds() / 60
        fig.add_vline(x=pred_t, line_color="green", line_dash="dot",
                      annotation_text="Fin predite", row=1, col=1)

    # Vrai dernier eclair
    if "is_last_lightning_cloud_ground" in sub.columns:
        last_true = sub[sub["is_last_lightning_cloud_ground"].astype(str) == "True"]
        if len(last_true) > 0:
            lt_min = (last_true.iloc[0]["date"] - t0).total_seconds() / 60
            fig.add_vline(x=lt_min, line_color="orange", line_dash="dashdot",
                          annotation_text="Vrai dernier", row=1, col=1)

    # Distance
    fig.add_trace(go.Scatter(
        x=sub["t_min"], y=sub["dist"],
        mode="lines", name="Distance",
        line=dict(color="#ff7f0e", width=2),
        fill="tozeroy", fillcolor="rgba(255,127,14,0.1)",
    ), row=2, col=1)
    fig.add_hline(y=3, line_dash="dot", line_color="red",
                  annotation_text="3 km (danger)", row=2, col=1)

    # Activite CG
    fig.add_trace(go.Bar(
        x=sub["t_min"], y=sub["n_cg_5min"],
        name="CG / 5 min", marker_color="#9467bd", opacity=0.7,
    ), row=3, col=1)

    fig.update_xaxes(title_text="Temps depuis debut alerte (min)", row=3, col=1)
    fig.update_yaxes(title_text="Score", range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="km", row=2, col=1)
    fig.update_yaxes(title_text="Nb CG", row=3, col=1)
    fig.update_layout(
        height=650, showlegend=False,
        margin=dict(t=40, b=20, l=60, r=20),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Resume textuel ────────────────────────────────────────────────────────
    if predicted_end:
        st.success(
            f"**Fin predite a {predicted_end.strftime('%H:%M:%S UTC')}** "
            f"(premier eclair avec score >= {theta:.0%}). "
            f"Gain estime : **{gain_min:.0f} minutes** vs la regle fixe de 30 minutes."
        )
    else:
        st.warning(
            f"Aucun eclair n'atteint le seuil theta = {theta:.0%}. "
            f"Essayez de baisser le seuil dans la barre laterale."
        )

    # ── Tableau detaille ──────────────────────────────────────────────────────
    with st.expander("Tableau detaille des predictions"):
        disp = sub[["date", "dist", "amplitude", "maxis", "confidence"]].copy()
        disp["score %"] = (disp["confidence"] * 100).round(1)
        disp["fin predite ?"] = disp["confidence"] >= theta
        disp["date"] = disp["date"].dt.strftime("%H:%M:%S")
        disp["dist"] = disp["dist"].round(2)
        disp = disp.drop(columns=["confidence"])
        st.dataframe(
            disp.style.background_gradient(subset=["score %"], cmap="RdYlGn"),
            use_container_width=True, height=350,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 : ANALYSE EXPLORATOIRE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Analyse exploratoire":

    st.header("Analyse exploratoire des donnees")
    st.markdown("Figures generees lors de l'EDA sur le dataset d'entrainement (2016-2022).")

    plots_dir = os.path.join(BASE, "plots")
    figures = [
        ("01_airport_analysis.png",       "Caracteristiques par aeroport"),
        ("02_temporal_analysis.png",       "Saisonnalite et patterns temporels"),
        ("03_alert_analysis.png",          "Structure des alertes"),
        ("04_spatial_analysis.png",        "Distribution spatiale (azimuth)"),
        ("05_last_lightning_analysis.png", "Signaux discriminants du dernier eclair"),
        ("06_alert_examples.png",          "Exemples d'alertes annotees"),
        ("06_correlation_matrix.png",      "Matrice de correlation"),
        ("07_feature_importance.png",      "Importance des features"),
        ("08_model_evaluation.png",        "Evaluation des modeles (ROC, PR)"),
        ("09_score_distribution.png",      "Distribution des scores de confiance"),
        ("10_evaluation_results.png",      "Resultats operationnels (Gain et Risque)"),
        ("11_eval_gain_risk.png",          "Courbe Gain-Risque vs theta"),
        ("12_model_v3_results.png",        "Resultats detailles V3"),
        ("13_comparison_final.png",        "Comparaison finale des approches"),
    ]

    for fname, title in figures:
        fpath = os.path.join(plots_dir, fname)
        if os.path.exists(fpath):
            st.subheader(title)
            st.image(fpath, use_container_width=True)
            st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 : GUIDE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Guide d'utilisation":

    st.header("Guide d'utilisation")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Demarrage rapide",
        "Interpreter les resultats",
        "Parametres",
        "Methodologie",
    ])

    with tab1:
        st.subheader("En 3 etapes")
        st.markdown("""
1. **Choisir une alerte** : selectionnez un aeroport et un numero d'alerte dans la page "Prediction"
2. **Ajuster theta** : utilisez le curseur dans la barre laterale (0.90 par defaut, recommande pour les aeroports)
3. **Lire le graphique** : la ligne bleue monte quand le modele detecte la fin de l'orage

Le modele calcule automatiquement les features et affiche le score de confiance pour chaque eclair CG.
        """)

        st.subheader("Format CSV attendu (si vous apportez vos propres donnees)")
        st.markdown("""
| Colonne | Type | Description |
|---|---|---|
| date | horodatage UTC | Date et heure de l'eclair |
| airport | texte | Nom de l'aeroport |
| airport_alert_id | entier | Identifiant de l'alerte |
| dist | km | Distance a l'aeroport |
| azimuth | degres | Direction (0-360) |
| amplitude | kA | Intensite du courant |
| maxis | kA | Courant crete |
| icloud | booleen | True = IC, False = CG |
        """)

    with tab2:
        st.subheader("Le score de confiance")
        st.markdown("""
Le score represente la probabilite estimee que l'eclair courant soit le dernier de l'alerte.
Il varie de 0 (orage en cours) a 1 (tres probablement la fin).

**Signaux captes par le modele :**
- Eclairs de plus en plus espaces (dt_prev_s croissant)
- Orage qui s'eloigne (distance croissante)
- Diminution de l'activite IC et CG
- Intensite (amplitude) qui declin
        """)

        st.subheader("Le seuil theta")
        st.markdown("""
Theta est le seuil de decision. Le premier eclair dont le score depasse theta declenche la levee d'alerte.

| Theta | Effet |
|---|---|
| Eleve (0.95) | Tres peu de fausses alarmes, gain limite |
| Moyen (0.85-0.90) | Bon compromis, recommande pour les aeroports |
| Bas (< 0.70) | Gain maximal mais risque eleve |
        """)

        st.subheader("Gain et risque")
        st.markdown("""
- **Gain** = temps recupere vs la regle fixe de 30 min
- **Risque** = proportion d'eclairs dangereux (< 3 km) survenant apres la fin predite
- **Contrainte** : Risque < 2% (recommandation Meteorage)
        """)

    with tab3:
        st.subheader("Seuils optimaux par aeroport")
        seuils = pd.DataFrame({
            "Aeroport": ["Ajaccio", "Bastia", "Biarritz", "Nantes", "Pise"],
            "Seuil F1-optimal": [0.387, 0.275, 0.224, 0.083, 0.528],
            "AUC": [0.931, 0.923, 0.923, 0.896, 0.929],
            "F1": [0.444, 0.384, 0.463, 0.407, 0.442],
            "Profil": [
                "Orages mediterraneens courts",
                "Orages mediterraneens courts",
                "Orages atlantiques variables",
                "Orages continentaux",
                "Orages liguriens, IC dominants",
            ],
        })
        st.dataframe(seuils, use_container_width=True, hide_index=True)
        st.info("Le seuil F1-optimal maximise le F1-score. Pour un usage operationnel (R < 2%), "
                "utiliser un theta plus eleve (0.85-0.95).")

    with tab4:
        st.subheader("Pipeline")
        st.code("""
Donnees brutes (CSV Meteorage)
    |
    v
Feature Engineering (67 features causales)
    |  - Temps inter-eclairs (dt_prev_s, moyennes glissantes, EMA)
    |  - Distance et tendance spatiale (dist_trend_W)
    |  - Amplitude et intensite (amp_abs, maxis)
    |  - Densite CG temporelle (n_cg_2/5/10/15/30min)
    |  - Contexte IC (n_ic, ratio_ic_cg, ic_trend)
    v
LightGBM (un modele par aeroport)
    |  class_weight='balanced', GroupKFold(5) par alerte
    v
Score de confiance [0, 1]
    |
    v
Seuil theta -> Decision de levee d'alerte
        """, language="text")

        st.subheader("Resultats de validation")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AUC", "0.9328")
        c2.metric("F1", "0.446")
        c3.metric("Gain (eval)", "213.8 h")
        c4.metric("Risque", "1.15%")

        st.subheader("Top 10 features")
        fi = pd.DataFrame({
            "Feature": ["azimuth", "dt_prev_s", "maxis", "dist_trend_3", "amp_abs",
                         "dist_trend_5", "dist_mean_3", "t_since_start_s", "dist", "rank_cg"],
            "Importance": [1.0, 0.99, 0.96, 0.82, 0.77, 0.77, 0.70, 0.66, 0.65, 0.65],
            "Categorie": ["Spatial", "Temporel", "Intensite", "Spatial", "Intensite",
                           "Spatial", "Spatial", "Temporel", "Spatial", "Temporel"],
        })
        fig_fi = px.bar(fi.sort_values("Importance"), x="Importance", y="Feature",
                        orientation="h", color="Categorie",
                        color_discrete_map={"Spatial": "#1f77b4", "Temporel": "#ff7f0e", "Intensite": "#9467bd"})
        fig_fi.update_layout(height=380)
        st.plotly_chart(fig_fi, use_container_width=True)
