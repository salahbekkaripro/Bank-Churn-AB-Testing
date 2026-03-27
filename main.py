import os
import json
import logging
import pandas as pd
import numpy as np
from src.stats_utils import run_statistical_test, estimate_business_value, plot_feature_importance
from src.model_trainer import train_churn_model

# Config pour afficher les messages dans la console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # 1. On charge le fichier de données
    data_path = 'data/Churn_Modelling.csv'
    if not os.path.exists(data_path):
        logger.error(f"Fichier non trouvé : {data_path}")
        return

    try:
        df = pd.read_csv(data_path)
        logger.info(f"Fichier chargé : {df.shape[0]} lignes.")
    except Exception as e:
        logger.error(f"Erreur de lecture : {e}")
        return

    # Nettoyage rapide des titres de colonnes
    df.columns = [c.strip() for c in df.columns]

    # 2. On entraîne le modèle pour voir qui risque de partir
    logger.info("Entraînement du modèle...")
    model, df, features = train_churn_model(df)
    
    # Création du graphique des variables importantes
    plot_feature_importance(model, features)

    # 3. On sélectionne les clients à risque (plus de 50% de chance de partir)
    # C'est sur eux qu'on va faire le test
    seuil_risque = 0.50
    df_cible = df[df['Churn_Probability'] > seuil_risque].copy()
    
    logger.info(f"Nombre de clients à risque (> {seuil_risque:.0%}) : {len(df_cible)} clients.")

    if len(df_cible) < 100:
        # Si on n'a pas assez de monde, on prend tout le monde pour que le test marche
        df_cible = df.copy()

    # 4. On sépare les clients en deux groupes (A et B) au hasard
    np.random.seed(42)
    df_cible['Group'] = np.random.choice(['A', 'B'], size=len(df_cible))

    # 5. Simulation : on imagine qu'on arrive à garder 15% des clients du groupe B
    mask_b_partis = (df_cible['Group'] == 'B') & (df_cible['Exited'] == 1)
    nb_partis_b = mask_b_partis.sum()
    
    # Calcul du nombre de clients qu'on sauve
    nb_sauves = int(nb_partis_b * 0.15)
    indices_sauves = df_cible[mask_b_partis].sample(n=nb_sauves, random_state=42).index
    
    # On met à jour les données pour simuler l'impact
    df_cible['Exited_Post'] = df_cible['Exited'].copy()
    df_cible.loc[indices_sauves, 'Exited_Post'] = 0
    logger.info(f"Simulation : {nb_sauves} clients sauvés dans le groupe B.")

    # 6. Calcul des statistiques
    stats_a = df_cible[df_cible['Group'] == 'A']['Exited_Post'].agg(['count', 'sum'])
    stats_b = df_cible[df_cible['Group'] == 'B']['Exited_Post'].agg(['count', 'sum'])
    
    z_stat, p_val, ci = run_statistical_test(
        success_a=int(stats_a['sum']),
        size_a=int(stats_a['count']),
        success_b=int(stats_b['sum']),
        size_b=int(stats_b['count'])
    )
    
    # 7. Calcul des résultats finaux
    taux_a = stats_a['sum'] / stats_a['count']
    taux_b = stats_b['sum'] / stats_b['count']
    reduction = (taux_a - taux_b) / taux_a
    
    gain_argent = estimate_business_value(df_cible, nb_sauves)
    
    logger.info(f"Résultats : Baisse du churn={reduction:.2%}, P-value={p_val:.4f}")

    # 8. On enregistre tout dans un fichier JSON pour l'affichage (Streamlit)
    resultats = {
        "metrics": {
            "churn_rate_control": float(taux_a),
            "churn_rate_test": float(taux_b),
            "lift": float(reduction)
        },
        "statistics": {
            "z_statistic": z_stat,
            "p_value": p_val,
            "confidence_intervals": ci,
            "is_significant": bool(p_val < 0.05)
        },
        "business_impact": {
            "customers_saved": int(nb_sauves),
            "estimated_financial_gain_euro": round(gain_argent, 2)
        },
        "ml_metadata": {
            "model": "XGBoost",
            "targeted_customers": len(df_cible),
            "risk_threshold": seuil_risque
        }
    }

    report_path = 'reports/ab_test_summary.json'
    os.makedirs('reports', exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(resultats, f, indent=4, ensure_ascii=False)
    
    logger.info(f"Rapport sauvegardé dans : {report_path}")

if __name__ == "__main__":
    main()
