# Data Battle 2026 - Prédiction de la Fin des Orages

## 👥 Équipe
- Nom de l’équipe : Pied Piper Jr.
- Membres :
  - HASSANA Mohamadou
  - KENGALI Pacome
  - KOMGUEM Helcias
  - MAFFO Natacha
  - MBASSI Loic
  - MOGOU Igor

## Description du Projet

Ce projet participe à la **Data Battle 2026** organisée par l'IA Pau, en collaboration avec Météorage. L'objectif principal est de développer un modèle probabiliste capable d'estimer la fin réelle d'un orage en analysant la dynamique spatio-temporelle des éclairs dans un rayon de 50 km autour de plusieurs aéroports.

### Contexte Métier
Les technologies actuelles permettent une excellente anticipation de l'arrivée d'un orage, mais déterminer son moment exact de fin reste complexe. Actuellement, les alertes restent actives pendant une durée fixe (30 à 60 minutes après le dernier éclair). Pour les secteurs critiques comme les aéroports, chaque minute compte pour reprendre les opérations.

### Problématique
- **Défi** : Estimer la probabilité qu'un éclair nuage-sol (CG) soit le dernier éclair d'une alerte orageuse.
- **Impact** : Permettre aux aéroports de reprendre leurs activités plus tôt, tout en maintenant un risque acceptable de nouveaux éclairs.

## Objectifs

1. **Prédiction probabiliste** : Développer un modèle capable de prédire si un éclair donné est le dernier de l'alerte.
2. **Analyse spatio-temporelle** : Exploiter les caractéristiques des éclairs (position, amplitude, direction, etc.).
3. **Évaluation rigoureuse** : Respecter les contraintes de risque (< 2%) tout en maximisant le gain temps.
4. **Analyse comparative** : Comparer les spécificités orageuses par aéroport.

## Données

### Sources
- **Météorage** : Fournisseur de données météorologiques et d'éclairs.
- **Période** : Données d'entraînement et d'évaluation couvrant plusieurs années.

### Structure des Données
| Colonne | Description |
|---|---|
| `lightning_id` | Identifiant unique global de l'éclair |
| `lightning_airport_id` | Identifiant de l'éclair pour un aéroport donné |
| `date` | Horodatage UTC de l'éclair |
| `lon`, `lat` | Coordonnées géographiques de l'éclair |
| `amplitude` | Amplitude du courant en kA |
| `maxis` | Courant max instantané (kA) |
| `icloud` | Indicateur éclair intra-nuage (True/False) |
| `dist` | Distance à l'aéroport (km, max ~30km) |
| `azimuth` | Direction depuis l'aéroport (degrés) |
| `airport` | Nom de l'aéroport |
| `airport_alert_id` | ID de l'alerte |
| `is_last_lightning_cloud_ground` | **CIBLE** : True si dernier éclair CG de l'alerte |

### Fichiers de Données
- `segment_alerts_all_airports_train/` : Données d'entraînement
- `segment_alerts_all_airports_eval.csv` : Données d'évaluation
- `dataset_test/` : Jeux de données de test et notebook d'évaluation

## Méthodologie

### 1. Exploration des Données (EDA)
- Analyse descriptive des éclairs par aéroport
- Visualisation spatio-temporelle des orages
- Identification des patterns et anomalies

### 2. Feature Engineering
- **Features temporelles** : Intervalles entre éclairs, tendances
- **Features spatiales** : Distance, azimuth, densité
- **Features physiques** : Amplitude, polarité, type d'éclair
- **36 features** sans fuite de données (causalité respectée)

### 3. Modélisation
- **Algorithme** : LightGBM (Gradient Boosting)
- **Validation** : Cross-validation GroupKFold(5) par alerte
- **Métriques** : AUC, Precision-Recall, Gain temps, Risque

### 4. Évaluation
- **Protocole officiel** : Gain temps et risque sur données d'évaluation
- **Seuil optimal** : θ = 0.95 pour équilibrer gain et risque

## Résultats

### Performances du Modèle
| Métrique | Valeur |
|---|---|
| AUC (Cross-val OOF) | **0.9328 ± 0.003** |
| Accuracy maximale | **95.4%** |
| Baseline Météorage | 80.77% |
| Gain temps (θ=0.95) | **213.8 heures** |
| Risque (θ=0.95) | **1.15%** (< 2% ✓) |

### Features Importantes
- `azimuth` : Direction de l'éclair
- `dt_prev_s` : Intervalle depuis l'éclair précédent
- `maxis` : Courant max instantané
- `dist_trend_3` : Tendance de distance (3 éclairs)
- `amp_abs` : Amplitude absolue

## Structure du Projet

```
Data_Battle_2026/
├── 01_EDA_exploration.ipynb          # Exploration et visualisation des données
├── 02_model_prediction.ipynb         # Modélisation et prédictions
├── CLAUDE.md                         # Instructions et avancement du projet
├── requirements.txt                  # Dépendances Python
├── README.md                         # Ce fichier
├── LICENSE                           # Licence du projet
├── models/                           # Modèles entraînés
│   └── lgbm_v2.pkl
├── plots/                            # Visualisations générées
├── predictions_*.csv                 # Prédictions sur train/eval
├── df_*.csv                          # DataFrames enrichis
├── segment_alerts_all_airports_train/ # Données d'entraînement
├── segment_alerts_all_airports_eval.csv # Données d'évaluation
└── dataset_test/                     # Tests et évaluation
    ├── dataset_set.csv
    └── Evaluation_databattle_meteorage.ipynb
```

## Installation et Utilisation

### Prérequis
- Python 3.8+
- Environnement virtuel recommandé

### Installation
```bash
# Cloner le repository (si applicable)
git clone https://github.com/PiedPiperJr/Data_Battle_2026
cd Data_Battle_2026

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Exécution
1. **EDA** : Ouvrir `01_EDA_exploration.ipynb` dans Jupyter
2. **Modélisation** : Exécuter `02_model_prediction.ipynb`
3. **Évaluation** : Utiliser `dataset_test/Evaluation_databattle_meteorage.ipynb`

## Prochaines Étapes

- [ ] Intégrer les éclairs IC (intra-nuage) comme contexte temporel
- [ ] Développer des modèles spécifiques par aéroport
- [ ] Améliorer les features de densité spatiale


## Contributeurs

- **Équipe Data Battle 2026** : Développement du modèle et analyses
- **Météorage** : Fourniture des données
- **IA Pau** : Organisation de la compétition
- **Pied Piper Jr** : Equipe competitirice

## Licence

Ce projet est sous licence [ Creative Commons CC0 1.0 Universal](LICENSE) - voir le fichier LICENSE pour plus de détails.

## 🔗 Liens Utiles

- [Site officiel Data Battle 2026](https://iapau.org/events/data-battle-2026/)


---

*Projet développé dans le cadre de la Data Battle 2026*