PROJET_CHURN_AB_TESTING/
├── data/
│   ├── raw/                # Données brutes (Churn_Modelling.csv)
│   └── processed/          # Données nettoyées et préparées
├── notebooks/
│   ├── 01_exploration.ipynb    # EDA (Analyse exploratoire)
│   └── 02_ab_testing.ipynb     # Simulation et tests statistiques
├── src/
│   ├── __init__.py
│   ├── preprocess.py       # Fonctions de nettoyage et encodage
│   ├── model.py            # Pipeline de prédiction (XGBoost/RF)
│   └── stats_utils.py      # Fonctions pour le Z-test et le Lift
├── reports/
│   ├── figures/            # Graphiques exportés (PNG/SVG)
│   └── GEMINI.md           # Rapport d'analyse et conclusions
├── requirements.txt        # Dépendances (pandas, statsmodels, etc.)
└── main.py                 # Point d'entrée pour exécuter le pipeline