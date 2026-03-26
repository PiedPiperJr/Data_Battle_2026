---
noteId: "c46ced4028e911f195b60326437f0554"
tags: []

---

⚡ Votre mission : prévoir l’évolution d’un orage, plus précisément d’estimer une probabilité de la fin de l’orage. En particulier, certaines zones sensibles, comme les aéroports, doivent surveiller de près la fin des orages afin de reprendre leur activité et chaque minute compte.

Les technologies actuelles permettent déjà une excellente anticipation de l’arrivée d’un orage. Toutefois, déterminer son moment exact de fin reste complexe. Aujourd’hui les alertes restent actives pendant une durée fixe, typiquement 30 à 60 minutes après l’occurrence du dernier éclair dans la zone de surveillance. Pour certains secteurs, notamment les aéroports, où chaque minute compte, cette méthode montre ses limites. Votre objectif sera donc de développer un modèle probabiliste capable d’estimer la fin réelle d’un orage, en analysant la dynamique spatio-temporelle des éclairs dans un rayon de 50 km autour de plusieurs aéroports européens.

Un deuxième axe est d’analyser les tendances d’orages pour chaque aéroport. En effet, chaque lieu a une condition géographique et météorologique particulière. Une analyse intéressante serait d’identifier les spécificités de chaque lieu et les principaux types d’orage existant dans les données.

À partir des données de meteorage, votre objectif sera donc de fournir les analyses permettant aux clients de meteorage de  subir la foudre le moins possible!


Plus d'information sur cette page : https://iapau.org/events/data-battle-2026/

les données d'entrainement se trouvent dans segment_alerts_all_airports_train/
les données d'évaluation se trouvent dans le fichier segment_alerts_all_airports_eval.csv
les données d'entrainement ainsi que le notebook pour asoocié se trouvent dans dataset_test/

pour poser des jalons sur le projet, on devra répondre à ces formulaires :
https://framaforms.org/questionnaire-semaine-1-data-battle-2026-ia-pau-1772737927
https://framaforms.org/questionnaire-semaine-2-data-battle-2026-ia-pau-1773420087
https://framaforms.org/questionnaire-semaine-3-data-battle-2026-ia-pau-1774030445

voici les modalités de livraison sont dans "Organisation rendus et soutenances Data Battle IA PAU 2026.docx"

Avant de commencer, ce serait bien parcourir des articles scientifiques qui traitent la question de prédictions des éclairs.

Puis extraire les colonnes du dataset et les informations générales sur les données, poser des hypothèses sur ce qui peut être exploités ou pas, effectuer une réduction de dimension si necessaire. un notebook de visualisation est indispensable 

NB: avec des moyens purement statistiques, d'après météorage, ils réussissent à prédire une alerte à 80,77% 

proposer des méthodes d'entrainement basées sur les précédentes analyses.

n'hésite pas à modifier CLAUDE.md pour mettre à jour les instructions si necessaire.


---

## Etat d'avancement (26 mars 2026)

### Travail effectue :
1. **EDA complete** : notebook `01_EDA_exploration.ipynb` + 20 figures dans `plots/`
2. **Feature engineering causal** (62 features sans fuite de donnees)
3. **Comparaison multi-modeles** : 10 configurations testees (LightGBM, XGBoost, RF, LogReg, Ensemble, Stacking)
4. **Cross-validation GroupKFold(5)** par alerte
5. **3 modeles finaux** : `lgbm_per_airport.pkl`, `lgbm_v3.pkl`, `xgb_tuned.pkl`
6. **Predictions eval** : `predictions.csv` (format officiel)
7. **Application Streamlit** : `app.py` (3 pages)
8. **Notebook complet** : `02_model_prediction.ipynb` (47 cellules, 18 images)
9. **README competition-ready** : jury comprend et teste en 2 minutes

### Resultats cles :
- Meilleur F1 (OOF) = **0.4489** (Ensemble)
- Meilleur AUC-ROC = **0.9365** (XGBoost tuned)
- Meilleur gain operationnel (R<2%) = **548.3h** (XGBoost, theta=0.85, R=1.76%)
- Per-airport : **573.8h** gain (theta=0.60, R=1.88%)
- Top features : azimuth, maxis, dist, dist_mean_3, amplitude

### Prochaines etapes :
- Formulaires de jalons a remplir
- Support de presentation PDF
- Lien de demonstration (video ou app deployee)
