import os
import json
import logging
import pandas as pd
import numpy as np
from src.stats_utils import run_statistical_test, estimate_business_value, plot_feature_importance
from src.model_trainer import train_churn_model

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # 1. Chargement des données
    data_path = 'data/Churn_Modelling.csv'
    if not os.path.exists(data_path):
        logger.error(f"Fichier non trouvé : {data_path}")
        return

    try:
        df = pd.read_csv(data_path)
        logger.info(f"Dataset chargé avec succès : {df.shape[0]} lignes.")
    except Exception as e:
        logger.error(f"Erreur lors de la lecture du CSV : {e}")
        return

    # Nettoyage des colonnes
    df.columns = [c.strip() for c in df.columns]

    # 2. Entraînement du modèle XGBoost
    logger.info("Entraînement du modèle XGBoost pour l'identification du risque...")
    model, df, features = train_churn_model(df)
    
    # Génération du graphique d'importance des features
    plot_feature_importance(model, features)

    # 3. Filtrage : Targeted A/B Testing
    # On effectue le test UNIQUEMENT sur les clients à haut risque (Probabilité > 50%)
    high_risk_threshold = 0.50
    df_targeted = df[df['Churn_Probability'] > high_risk_threshold].copy()
    
    logger.info(f"Identification des clients à haut risque (> {high_risk_threshold:.0%}) : {len(df_targeted)} clients.")

    if len(df_targeted) < 100:
        logger.warning("Nombre de clients à haut risque trop faible pour un A/B test robuste. Utilisation de tout le dataset.")
        df_targeted = df.copy()

    # 4. Assignation des groupes A/B sur les clients ciblés
    np.random.seed(42)
    df_targeted['Group'] = np.random.choice(['A', 'B'], size=len(df_targeted))
    logger.info("Groupes A/B assignés aux clients à haut risque.")

    # 5. Simulation de l'impact (Réduction de churn de 15% sur le groupe B)
    mask_b_exited = (df_targeted['Group'] == 'B') & (df_targeted['Exited'] == 1)
    n_potential_churn_b = mask_b_exited.sum()
    
    # On en sauve 15%
    n_saved = int(n_potential_churn_b * 0.15)
    indices_to_save = df_targeted[mask_b_exited].sample(n=n_saved, random_state=42).index
    
    df_targeted['Exited_Post'] = df_targeted['Exited'].copy()
    df_targeted.loc[indices_to_save, 'Exited_Post'] = 0
    logger.info(f"Simulation : {n_saved} clients sauvés sur {n_potential_churn_b} churners potentiels dans le groupe B.")

    # 6. Analyse Statistique
    stats_a = df_targeted[df_targeted['Group'] == 'A']['Exited_Post'].agg(['count', 'sum'])
    stats_b = df_targeted[df_targeted['Group'] == 'B']['Exited_Post'].agg(['count', 'sum'])
    
    z_stat, p_val, ci = run_statistical_test(
        success_a=int(stats_a['sum']),
        size_a=int(stats_a['count']),
        success_b=int(stats_b['sum']),
        size_b=int(stats_b['count'])
    )
    
    # 7. Calcul des métriques métier
    churn_rate_a = stats_a['sum'] / stats_a['count']
    churn_rate_b = stats_b['sum'] / stats_b['count']
    lift = (churn_rate_a - churn_rate_b) / churn_rate_a
    
    financial_impact = estimate_business_value(df_targeted, n_saved)
    
    logger.info(f"Résultats Ciblés : Lift={lift:.2%}, P-value={p_val:.4f}")

    # 8. Sauvegarde du rapport JSON (Même format pour compatibilité app.py)
    report = {
        "metrics": {
            "churn_rate_control": float(churn_rate_a),
            "churn_rate_test": float(churn_rate_b),
            "lift": float(lift)
        },
        "statistics": {
            "z_statistic": z_stat,
            "p_value": p_val,
            "confidence_intervals": ci,
            "is_significant": bool(p_val < 0.05)
        },
        "business_impact": {
            "customers_saved": int(n_saved),
            "estimated_financial_gain_euro": round(financial_impact, 2)
        },
        "ml_metadata": {
            "model": "XGBoost Classifier",
            "targeted_customers": len(df_targeted),
            "risk_threshold": high_risk_threshold
        }
    }

    report_path = 'reports/ab_test_summary.json'
    os.makedirs('reports', exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    
    logger.info(f"Rapport JSON mis à jour (Targeted A/B Test) : {report_path}")

if __name__ == "__main__":
    main()
