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
        "Tester / Predire",
        "Analyse exploratoire",
        "Guide d'utilisation",
    ])
    st.divider()

    theta = st.slider("Seuil de decision (theta)", 0.50, 0.99, 0.90, 0.01,
                       help="Premier eclair avec score >= theta declenchera la levee d'alerte. "
                            "Plus theta est eleve, moins de fausses alarmes mais gain reduit.")

    st.divider()
    st.markdown("**Performances (eval 2023-2025)**")
    st.markdown(f"- Gain : **548.3 h** sur 1081 alertes")
    st.markdown(f"- Risque : **1.76%** (< 2%)")
    st.markdown(f"- ~30 min recuperees par alerte")


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
    test_path = os.path.join(BASE, "dataset_test", "dataset_set.csv")
    datasets = {}
    if os.path.exists(test_path):
        datasets["Test set (dataset_test)"] = test_path
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
# PAGE 2 : TESTER / PREDIRE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Tester / Predire":

    st.header("Tester une prediction")
    st.markdown("Uploadez votre propre fichier CSV ou saisissez les parametres manuellement.")

    mode = st.radio("Mode", ["Upload CSV", "Saisie manuelle (un eclair)", "Demo rapide (dataset eval)"],
                    horizontal=True)

    if mode == "Upload CSV":
        st.subheader("Charger un fichier CSV")
        st.info(
            "Le fichier doit contenir les colonnes : **date**, **airport**, **airport_alert_id**, "
            "**dist**, **azimuth**, **amplitude**, **maxis**, **icloud**. "
            "Les eclairs CG doivent avoir icloud=False."
        )
        uploaded = st.file_uploader("Fichier CSV", type=["csv"])
        if uploaded is not None:
            try:
                df_up = pd.read_csv(uploaded)
                df_up["date"] = pd.to_datetime(df_up["date"], utc=True)
                df_up["icloud"] = df_up["icloud"].map(
                    {"True": True, "False": False, True: True, False: False}
                ).fillna(df_up["icloud"]).astype(bool)

                st.success(f"Fichier charge : {len(df_up)} lignes, {df_up['airport'].nunique()} aeroport(s), "
                           f"{df_up['airport_alert_id'].nunique()} alerte(s)")

                # Selection aeroport / alerte
                col1, col2 = st.columns(2)
                with col1:
                    ap_up = st.selectbox("Aeroport", sorted(df_up["airport"].unique()), key="up_ap")
                with col2:
                    alerts_up = sorted(df_up[(df_up["airport"] == ap_up) & (~df_up["icloud"])]["airport_alert_id"].unique())
                    al_up = st.selectbox("Alerte", alerts_up, key="up_al")

                df_alert_up = df_up[(df_up["airport"] == ap_up) & (df_up["airport_alert_id"] == al_up)]
                df_cg_up = df_alert_up[~df_alert_up["icloud"]].copy()

                if len(df_cg_up) == 0:
                    st.warning("Aucun eclair CG dans cette alerte.")
                else:
                    with st.spinner("Calcul des features et prediction..."):
                        df_feat_up = make_features(df_cg_up.copy(), df_up)
                        scores_up = predict(df_feat_up, models, ap_up)
                        df_feat_up["confidence"] = scores_up

                    sub_up = df_feat_up.sort_values("date").reset_index(drop=True)
                    t0_up = sub_up["date"].min()
                    sub_up["t_min"] = (sub_up["date"] - t0_up).dt.total_seconds() / 60

                    pred_above_up = sub_up[sub_up["confidence"] >= theta]
                    predicted_end_up = pred_above_up["date"].min() if len(pred_above_up) > 0 else None
                    last_date_up = sub_up["date"].max()
                    baseline_end_up = last_date_up + pd.Timedelta(minutes=30)
                    gain_up = (baseline_end_up - predicted_end_up).total_seconds() / 60 if predicted_end_up else 0

                    # KPIs
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Eclairs CG", len(sub_up))
                    k2.metric("Score max", f"{sub_up['confidence'].max():.1%}")
                    if predicted_end_up:
                        k3.metric("Fin predite", predicted_end_up.strftime("%H:%M:%S UTC"))
                        k4.metric("Gain estime", f"{gain_up:.0f} min")
                    else:
                        k3.metric("Fin predite", "Non atteint")
                        k4.metric("Gain estime", "0 min")

                    # Chart
                    fig_up = go.Figure()
                    fig_up.add_trace(go.Scatter(
                        x=sub_up["t_min"], y=sub_up["confidence"],
                        mode="lines+markers", name="Score",
                        line=dict(color="#1f77b4", width=2.5), marker=dict(size=5),
                    ))
                    fig_up.add_hline(y=theta, line_dash="dash", line_color="red",
                                     annotation_text=f"theta = {theta:.2f}")
                    if predicted_end_up:
                        pred_t_up = (predicted_end_up - t0_up).total_seconds() / 60
                        fig_up.add_vline(x=pred_t_up, line_color="green", line_dash="dot",
                                         annotation_text="Fin predite")
                    fig_up.update_layout(
                        title="Score de confiance par eclair CG",
                        xaxis_title="Temps depuis debut alerte (min)",
                        yaxis_title="Score", yaxis_range=[0, 1.05],
                        height=400, plot_bgcolor="white",
                    )
                    st.plotly_chart(fig_up, use_container_width=True)

                    # Detailed table
                    with st.expander("Tableau detaille"):
                        disp_up = sub_up[["date", "dist", "amplitude", "maxis", "azimuth", "confidence"]].copy()
                        disp_up["score %"] = (disp_up["confidence"] * 100).round(1)
                        disp_up["fin predite ?"] = disp_up["confidence"] >= theta
                        disp_up["date"] = disp_up["date"].dt.strftime("%H:%M:%S")
                        disp_up = disp_up.drop(columns=["confidence"])
                        st.dataframe(disp_up, use_container_width=True, height=300)

            except Exception as e:
                st.error(f"Erreur lors du chargement : {e}")

    elif mode == "Saisie manuelle (un eclair)":
        st.subheader("Simuler un eclair unique")
        st.markdown("Entrez les parametres d'un eclair fictif pour voir le score du modele.")

        col1, col2, col3 = st.columns(3)
        with col1:
            man_airport = st.selectbox("Aeroport", ["Ajaccio", "Bastia", "Biarritz", "Nantes", "Pise"], key="man_ap")
            man_dist = st.number_input("Distance (km)", 0.0, 50.0, 8.0, 0.5)
            man_azimuth = st.number_input("Azimuth (degres)", 0.0, 360.0, 180.0, 5.0)
        with col2:
            man_amplitude = st.number_input("Amplitude (kA)", -200.0, 200.0, -15.0, 1.0)
            man_maxis = st.number_input("Maxis (kA)", 0.0, 300.0, 20.0, 1.0)
            man_month = st.selectbox("Mois", list(range(1, 13)), index=6)
        with col3:
            man_hour = st.selectbox("Heure UTC", list(range(0, 24)), index=15)
            man_rank = st.number_input("Rang CG dans l'alerte", 1, 500, 10)
            man_dt_prev = st.number_input("Temps depuis eclair precedent (s)", 0.0, 3600.0, 60.0, 10.0)

        if st.button("Predire", type="primary"):
            # Build a minimal feature vector
            feat_dict = {
                "rank_cg": man_rank,
                "t_since_start_s": man_rank * man_dt_prev,
                "dt_prev_s": man_dt_prev,
                "amp_abs": abs(man_amplitude),
                "amp_sign": np.sign(man_amplitude),
                "dist": man_dist,
                "azimuth": man_azimuth,
                "amplitude": man_amplitude,
                "maxis": man_maxis,
                "month": man_month,
                "hour": man_hour,
            }
            # Fill rolling window features with approximate values
            for w in [3, 5, 10, 20]:
                feat_dict[f"dt_mean_{w}"] = man_dt_prev
                feat_dict[f"dt_max_{w}"] = man_dt_prev * 1.5
                feat_dict[f"dt_std_{w}"] = man_dt_prev * 0.3
                feat_dict[f"dt_ema_{w}"] = man_dt_prev
                feat_dict[f"dist_mean_{w}"] = man_dist
                feat_dict[f"dist_trend_{w}"] = 0.0
                feat_dict[f"amp_mean_{w}"] = abs(man_amplitude)
                feat_dict[f"amp_trend_{w}"] = 0.0
            for wm in [2, 5, 10, 15, 30]:
                feat_dict[f"n_cg_{wm}min"] = max(1, int(wm * 60 / max(man_dt_prev, 1)))
            feat_dict["rate_decline_cg"] = feat_dict["n_cg_5min"] / (feat_dict["n_cg_10min"] + 1)
            for c in ["n_ic_2min", "n_ic_5min", "n_ic_10min", "n_ic_30min",
                       "n_all_2min", "n_all_5min", "n_all_10min"]:
                feat_dict[c] = 0
            feat_dict["ratio_ic_cg_5min"] = 0.0
            feat_dict["ratio_ic_cg_10min"] = 0.0
            feat_dict["ic_trend"] = 0.0
            feat_dict["total_activity"] = feat_dict["n_all_5min"]
            feat_dict["activity_decline"] = 0.5

            df_man = pd.DataFrame([feat_dict])
            score_man = predict(df_man, models, man_airport)
            score_val = score_man[0]

            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Score de confiance", f"{score_val:.1%}")
            m2.metric("Seuil theta", f"{theta:.0%}")
            if score_val >= theta:
                m3.metric("Decision", "LEVER L'ALERTE", delta="Score >= theta")
                st.success(f"Le modele predit que cet eclair est probablement le dernier (score={score_val:.3f} >= theta={theta:.2f}).")
            else:
                m3.metric("Decision", "MAINTENIR L'ALERTE", delta="Score < theta", delta_color="inverse")
                st.warning(f"L'orage est probablement encore actif (score={score_val:.3f} < theta={theta:.2f}).")

            # Score gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score_val * 100,
                number={"suffix": "%"},
                title={"text": "Score de confiance"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#1f77b4"},
                    "steps": [
                        {"range": [0, theta * 100], "color": "#ffebee"},
                        {"range": [theta * 100, 100], "color": "#e8f5e9"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": theta * 100,
                    },
                },
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.caption("Note : la saisie manuelle est une approximation. Les features de fenetre glissante "
                       "sont estimees a partir d'un seul eclair. Pour des resultats precis, "
                       "uploadez un CSV complet avec la sequence d'eclairs.")

    else:  # Demo rapide
        st.subheader("Demo rapide")
        st.markdown("Choisir un dataset et un aeroport, puis cliquer sur **Alerte aleatoire**.")

        demo_datasets = {}
        test_p = os.path.join(BASE, "dataset_test", "dataset_set.csv")
        eval_p = os.path.join(BASE, "segment_alerts_all_airports_eval.csv")
        if os.path.exists(test_p):
            demo_datasets["Test set (dataset_test)"] = test_p
        if os.path.exists(eval_p):
            demo_datasets["Evaluation (2023-2025)"] = eval_p

        if not demo_datasets:
            st.error("Aucun dataset trouve.")
            st.stop()

        demo_ds = st.selectbox("Dataset", list(demo_datasets.keys()), key="demo_ds")
        df_eval = load_dataset(demo_datasets[demo_ds])
        labeled_eval = df_eval[df_eval["airport_alert_id"].notnull()]

        demo_ap = st.selectbox("Aeroport", sorted(labeled_eval["airport"].unique()), key="demo_ap")
        demo_alerts = sorted(labeled_eval[labeled_eval["airport"] == demo_ap]["airport_alert_id"].unique())

        if st.button("Alerte aleatoire", type="primary"):
            st.session_state["demo_alert"] = int(np.random.choice(demo_alerts))

        demo_alert = st.session_state.get("demo_alert", demo_alerts[0] if demo_alerts else None)
        if demo_alert is not None:
            st.info(f"Alerte **#{int(demo_alert)}** a **{demo_ap}**")

            df_demo_full = labeled_eval[(labeled_eval["airport"] == demo_ap) &
                                         (labeled_eval["airport_alert_id"] == demo_alert)]
            df_demo_cg = df_demo_full[~df_demo_full["icloud"]].copy()

            if len(df_demo_cg) == 0:
                st.warning("Aucun eclair CG.")
            else:
                with st.spinner("Prediction en cours..."):
                    df_demo_feat = make_features(df_demo_cg.copy(), df_eval)
                    scores_demo = predict(df_demo_feat, models, demo_ap)
                    df_demo_feat["confidence"] = scores_demo

                sub_demo = df_demo_feat.sort_values("date").reset_index(drop=True)
                t0_d = sub_demo["date"].min()
                sub_demo["t_min"] = (sub_demo["date"] - t0_d).dt.total_seconds() / 60

                last_date_d = sub_demo["date"].max()
                baseline_end_d = last_date_d + pd.Timedelta(minutes=30)
                pa_d = sub_demo[sub_demo["confidence"] >= theta]
                pe_d = pa_d["date"].min() if len(pa_d) > 0 else None
                gain_d = (baseline_end_d - pe_d).total_seconds() / 60 if pe_d else 0

                # KPIs
                dk1, dk2, dk3, dk4 = st.columns(4)
                dk1.metric("Eclairs CG", len(sub_demo))
                dk2.metric("Duree", f"{(last_date_d - t0_d).total_seconds()/60:.0f} min")
                dk3.metric("Score max", f"{sub_demo['confidence'].max():.1%}")
                dk4.metric("Gain estime", f"{gain_d:.0f} min" if pe_d else "0 min")

                # 3-panel chart
                fig_demo = make_subplots(
                    rows=3, cols=1, shared_xaxes=True,
                    row_heights=[0.45, 0.30, 0.25],
                    subplot_titles=["Score de confiance", "Distance (km)", "Activite CG (5 min)"],
                    vertical_spacing=0.08,
                )
                fig_demo.add_trace(go.Scatter(
                    x=sub_demo["t_min"], y=sub_demo["confidence"],
                    mode="lines+markers", name="Score",
                    line=dict(color="#1f77b4", width=2.5), marker=dict(size=5),
                ), row=1, col=1)
                fig_demo.add_hline(y=theta, line_dash="dash", line_color="red",
                                   annotation_text=f"theta={theta:.2f}", row=1, col=1)
                if pe_d:
                    fig_demo.add_vline(x=(pe_d - t0_d).total_seconds()/60,
                                       line_color="green", line_dash="dot",
                                       annotation_text="Fin predite", row=1, col=1)
                if "is_last_lightning_cloud_ground" in sub_demo.columns:
                    lt_d = sub_demo[sub_demo["is_last_lightning_cloud_ground"].astype(str) == "True"]
                    if len(lt_d) > 0:
                        fig_demo.add_vline(x=(lt_d.iloc[0]["date"] - t0_d).total_seconds()/60,
                                           line_color="orange", line_dash="dashdot",
                                           annotation_text="Vrai dernier", row=1, col=1)

                fig_demo.add_trace(go.Scatter(
                    x=sub_demo["t_min"], y=sub_demo["dist"],
                    mode="lines", name="Distance",
                    line=dict(color="#ff7f0e", width=2), fill="tozeroy",
                    fillcolor="rgba(255,127,14,0.1)",
                ), row=2, col=1)
                fig_demo.add_hline(y=3, line_dash="dot", line_color="red",
                                   annotation_text="3 km", row=2, col=1)

                fig_demo.add_trace(go.Bar(
                    x=sub_demo["t_min"], y=sub_demo["n_cg_5min"],
                    name="CG/5min", marker_color="#9467bd", opacity=0.7,
                ), row=3, col=1)

                fig_demo.update_xaxes(title_text="Temps (min)", row=3, col=1)
                fig_demo.update_yaxes(range=[0, 1.05], row=1, col=1)
                fig_demo.update_layout(height=650, showlegend=False,
                                       plot_bgcolor="white", paper_bgcolor="white",
                                       margin=dict(t=40, b=20, l=60, r=20))
                st.plotly_chart(fig_demo, use_container_width=True)

                if pe_d:
                    st.success(f"Fin predite a **{pe_d.strftime('%H:%M:%S UTC')}** — Gain : **{gain_d:.0f} min**")
                else:
                    st.warning("Seuil non atteint. Essayez de baisser theta.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 : ANALYSE EXPLORATOIRE
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
        ("14_model_comparison.png",        "Comparaison multi-modeles (F1, AUC, PR)"),
        ("15_precision_recall_curves.png", "Courbes Precision-Recall"),
        ("16_roc_curves.png",              "Courbes ROC"),
        ("17_feature_importance_final.png","Importance des features (finale)"),
        ("18_operational_evaluation.png",  "Evaluation operationnelle (Gain/Risque vs theta)"),
        ("19_gain_vs_risk.png",            "Compromis Gain vs Risque"),
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
Feature Engineering (62 features causales)
    |  - Temps inter-eclairs (dt_prev_s, moyennes glissantes, EMA)
    |  - Distance et tendance spatiale (dist_trend_W)
    |  - Amplitude et intensite (amp_abs, maxis)
    |  - Densite CG temporelle (n_cg_2/5/10/15/30min)
    |  - Contexte IC (n_ic, ratio_ic_cg, ic_trend)
    v
Multi-modeles (GroupKFold 5 par alerte)
    |  - XGBoost tuned (1500 trees) → AUC 0.937
    |  - LightGBM tuned (2000 trees) → F1 0.440
    |  - LightGBM per-airport → Gain 573.8h
    |  - Ensemble (moyenne des 3)  → F1 0.449
    v
Score de confiance [0, 1]
    |
    v
Seuil theta (0.85 recommande) → Decision de levee d'alerte
        """, language="text")

        st.subheader("Resultats de validation")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AUC", "0.9365")
        c2.metric("F1", "0.449")
        c3.metric("Gain (eval)", "548.3 h")
        c4.metric("Risque", "1.76%")

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
