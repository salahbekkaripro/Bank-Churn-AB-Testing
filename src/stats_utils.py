import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

def run_statistical_test(success_a, size_a, success_b, size_b, alpha=0.05):
    """
    On compare les taux de départ entre le groupe A et le groupe B.
    """
    count = np.array([success_b, success_a])
    nobs = np.array([size_b, size_a])
    
    # Test statistique pour voir si le groupe B part moins que le A
    z_stat, p_value = proportions_ztest(count, nobs, alternative='smaller')
    
    # Calcul des intervalles de confiance
    ci_a = proportion_confint(success_a, size_a, alpha=alpha, method='normal')
    ci_b = proportion_confint(success_b, size_b, alpha=alpha, method='normal')
    
    return float(z_stat), float(p_value), {"group_a": ci_a, "group_b": ci_b}

def estimate_business_value(df, n_saved, target_col='EstimatedSalary'):
    """
    On estime combien d'argent on garde en sauvant ces clients (en utilisant leur salaire moyen).
    """
    avg_value = df[df['Exited'] == 0][target_col].mean()
    return float(n_saved * avg_value)

def plot_feature_importance(model, feature_names):
    """
    Affiche un graphique des variables qui expliquent le plus le départ des clients.
    """
    plt.figure(figsize=(10, 8))
    
    # Récupération de l'importance des variables
    importance = model.feature_importances_
    
    # On met ça dans un tableau pour l'affichage
    feat_imp = pd.DataFrame({
        'Variable': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    # On utilise Seaborn pour faire un joli graphique
    sns.set_style("whitegrid")
    sns.barplot(x='Importance', y='Variable', data=feat_imp, palette='magma')
    
    plt.title('Pourquoi les clients partent ? (Importance des variables)', fontsize=15)
    plt.tight_layout()
    
    # Sauvegarde du graphique
    plt.savefig('reports/feature_importance.png')
    plt.close()
    print("Graphique sauvegardé dans 'reports/feature_importance.png'.")
