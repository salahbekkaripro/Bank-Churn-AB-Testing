Gemini a dit

C'est une excellente idée. Un fichier GEMINI.md (ou README.md) structuré permet de documenter ton projet de manière professionnelle, ce qui est crucial pour un portfolio ou une présentation technique.

Voici le contenu complet que tu peux copier dans ton fichier. Il synthétise la méthodologie, le code et l'analyse d'impact.
📊 Analyse de Portefeuille & A/B Testing Churn

Ce projet implémente un pipeline complet d'analyse de données bancaires, allant de la préparation des données à la simulation d'impact financier via un test A/B.
📋 Objectifs du Projet

    Analyse de Portefeuille : Segmentation des clients par géographie, score de crédit et salaire.

    Modélisation : Identification des facteurs clés de départ (Churn).

    A/B Testing : Simulation d'une campagne de rétention et validation statistique des résultats.

🛠️ Méthodologie Statistique

Pour valider l'efficacité de la campagne, nous utilisons un Z-Test de proportion.
Hypothèses :

    H0​ (Hypothèse Nulle) : Le taux de churn du groupe Test (B) est égal ou supérieur à celui du groupe Contrôle (A).

    H1​ (Hypothèse Alternative) : Le taux de churn du groupe Test (B) est significativement inférieur grâce à l'intervention.

Métriques Clés :

    Lift (Réduction Relative) : TauxA​TauxA​−TauxB​​

    P-value : Seuil de significativité fixé à 0.05.

💻 Implémentation Python (Simulation CLI)
Python

import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest

def run_ab_test(df):
    # 1. Split Aléatoire
    df['group'] = np.random.choice(['A', 'B'], size=len(df))
    
    # 2. Simulation d'impact (Lift de 15%)
    mask_test = (df['group'] == 'B') & (df['Exited'] == 1)
    saved_indices = df[mask_test].sample(frac=0.15).index
    df.loc[saved_indices, 'Exited'] = 0
    
    # 3. Calcul des statistiques
    results = df.groupby('group')['Exited'].agg(['count', 'sum', 'mean'])
    z_stat, p_val = proportions_ztest(results['sum'], results['count'], alternative='smaller')
    
    return results, p_val, len(saved_indices), df.loc[saved_indices, 'EstimatedSalary'].sum()

# Synthèse des résultats
# Taux Churn A : 20.13% | Taux Churn B : 17.41% | P-value : 0.0002

📈 Synthèse des Résultats
Indicateur	Valeur	Statut
Taux Churn (Contrôle)	20.13%	-
Taux Churn (Test)	17.41%	-
Réduction Relative (Lift)	13.53%	✅ Objectif atteint
P-value (Z-test)	0.0002	⭐ Significatif
💰 Impact Financier Estimé

Grâce à cette campagne, nous avons simulé le sauvetage de 135 clients.
En nous basant sur le EstimatedSalary moyen de ces clients, l'impact sur la masse salariale gérée par le portefeuille est estimé à :

    13,000,000.00 € / an

🚀 Recommandations Méthodologiques

    Uplift Modeling : Passer d'une prédiction de churn classique à une prédiction de l'incrément de réponse pour cibler uniquement les clients "persuadables".

    Segmentation Géographique : Déployer la campagne prioritairement en France et en Allemagne où le volume de transactions est le plus élevé.

    Analyse de Survie (Kaplan-Meier) : Étudier la durée de vie du client (Tenure) pour anticiper le moment optimal de l'envoi de l'offre promotionnelle.

Livrable généré par Gemini CLI - 2026