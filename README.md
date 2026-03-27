# Analyse du Churn et Simulation d'A/B Test

Ce projet a été réalisé dans le cadre d'un cours de Licence 3 (Informatique / Data Science). L'objectif est d'analyser les départs de clients dans une banque et de simuler l'impact d'une campagne de rétention via un test A/B.

## Présentation du projet
Le projet utilise un jeu de données de 10 000 clients bancaires. Nous cherchons à :
1. Identifier les clients les plus susceptibles de partir (Churn).
2. Simuler une offre promotionnelle sur un groupe de test.
3. Valider statistiquement si l'offre permet de réduire significativement le taux de départ.

## Résultats de l'analyse
D'après nos simulations, une réduction de 15% du churn dans le groupe de test permet de :
- Sauver environ 150 clients sur un cycle.
- Réduire le taux de départ de manière significative (validation par un test de proportion).
- Identifier les facteurs principaux du départ : le nombre de produits possédés par le client et son niveau d'activité.

## Structure du dossier
- `data/` : Contient le fichier CSV des clients.
- `src/` : Code source pour le modèle et les calculs statistiques.
- `reports/` : Résultats générés au format JSON et graphiques.
- `main.py` : Script principal pour lancer l'analyse.
- `app.py` : Dashboard pour visualiser les résultats.

## Installation et lancement

### 1. Installation des bibliothèques
Il est nécessaire d'installer les dépendances suivantes :
```bash
pip install pandas numpy xgboost scikit-learn matplotlib seaborn statsmodels streamlit plotly
```

### 2. Lancer l'analyse
Pour calculer les résultats et générer le rapport :
```bash
python main.py
```

### 3. Voir le dashboard
Pour lancer l'interface de visualisation :
```bash
streamlit run app.py
```

---
Projet universitaire - 2026
