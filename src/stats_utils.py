import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

def run_statistical_test(success_a, size_a, success_b, size_b, alpha=0.05):
    """
    Exécute un Z-test de proportion et calcule les intervalles de confiance.
    """
    count = np.array([success_b, success_a])
    nobs = np.array([size_b, size_a])
    
    # Z-test (H1: proportion B < proportion A)
    z_stat, p_value = proportions_ztest(count, nobs, alternative='smaller')
    
    # Intervalles de confiance
    ci_a = proportion_confint(success_a, size_a, alpha=alpha, method='normal')
    ci_b = proportion_confint(success_b, size_b, alpha=alpha, method='normal')
    
    return float(z_stat), float(p_value), {"group_a": ci_a, "group_b": ci_b}

def estimate_business_value(df, n_saved, target_col='EstimatedSalary'):
    """
    Estime le gain financier basé sur le salaire moyen des clients sauvés.
    """
    avg_value = df[df['Exited'] == 0][target_col].mean()
    return float(n_saved * avg_value)

def plot_feature_importance(model, feature_names):
    """
    Génère un graphique des facteurs qui causent le départ (Feature Importance).
    """
    plt.figure(figsize=(10, 8))
    
    # Extraction de l'importance
    importance = model.feature_importances_
    
    # Création du DataFrame pour plotting
    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    # Visualisation Seaborn
    sns.set_style("whitegrid")
    sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='magma')
    
    plt.title('Facteurs Clés du Churn (XGBoost Feature Importance)', fontsize=15)
    plt.tight_layout()
    
    # Sauvegarde
    plt.savefig('reports/feature_importance.png')
    plt.close()
    print("Graphique d'importance des features sauvegardé dans 'reports/feature_importance.png'.")
