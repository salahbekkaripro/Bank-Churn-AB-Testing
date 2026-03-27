import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

# Set seed for reproducibility
np.random.seed(42)

def load_and_preprocess(filepath):
    """Charge le dataset et nettoie les colonnes."""
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset chargé : {df.shape[0]} lignes.")
    except FileNotFoundError:
        print("Fichier non trouvé, création d'un dummy dataframe pour l'exemple.")
        data = {
            'RowNumber': range(1, 1001),
            'CustomerId': np.random.randint(15000000, 16000000, 1000),
            'Surname': ['Surname']*1000,
            'CreditScore': np.random.randint(300, 850, 1000),
            'Geography': np.random.choice(['France', 'Spain', 'Germany'], 1000),
            'Gender': np.random.choice(['Male', 'Female'], 1000),
            'Age': np.random.randint(18, 80, 1000),
            'Tenure': np.random.randint(0, 11, 1000),
            'Balance': np.random.uniform(0, 200000, 1000),
            'NumOfProducts': np.random.randint(1, 5, 1000),
            'HasCrCard': np.random.randint(0, 2, 1000),
            'IsActiveMember': np.random.randint(0, 2, 1000),
            'EstimatedSalary': np.random.uniform(10000, 150000, 1000),
            'Exited': np.random.choice([0, 1], 1000, p=[0.8, 0.2])
        }
        df = pd.DataFrame(data)
    
    # Nettoyage des noms de colonnes (snake_case optionnel, ici on garde l'original mais strip)
    df.columns = [c.strip() for c in df.columns]
    return df

def run_ab_test_simulation(df):
    """Simule le split A/B et l'intervention sur le groupe B."""
    
    # 1. Split A/B (50/50)
    df['Group'] = np.random.choice(['A', 'B'], size=len(df))
    
    # 2. Simulation d'Intervention sur le Groupe B
    # On réduit la probabilité de churn de 15% par rapport au churn initial
    # Si Exited == 1, on a 15% de chances qu'il devienne 0 dans le groupe B
    def simulate_churn(row):
        if row['Group'] == 'B' and row['Exited'] == 1:
            # 15% de réduction du churn
            if np.random.rand() < 0.15:
                return 0
        return row['Exited']

    df['Exited_Post'] = df.apply(simulate_churn, axis=1)
    return df

def analyze_results(df):
    """Calcule les statistiques et effectue le Z-test."""
    
    results = df.groupby('Group')['Exited_Post'].agg(['count', 'sum', 'mean']).rename(
        columns={'sum': 'churns', 'count': 'total', 'mean': 'churn_rate'}
    )
    
    n_a = results.loc['A', 'total']
    n_b = results.loc['B', 'total']
    c_a = results.loc['A', 'churns']
    c_b = results.loc['B', 'churns']
    
    # Z-test pour proportions
    count = np.array([c_b, c_a])
    nobs = np.array([n_b, n_a])
    z_stat, p_val = proportions_ztest(count, nobs, alternative='smaller')
    
    # Intervalles de confiance (95%)
    ci_b = proportion_confint(c_b, n_b, alpha=0.05, method='normal')
    ci_a = proportion_confint(c_a, n_a, alpha=0.05, method='normal')
    
    return results, z_stat, p_val, (ci_a, ci_b)

def visualize_results(results, ci):
    """Génère la visualisation Seaborn."""
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Bar plot
    ax = sns.barplot(x=results.index, y=results['churn_rate'], hue=results.index, palette='viridis', legend=False)
    
    # Ajout des barres d'erreur manuelles (IC 95%)
    y_err = [
        results.loc['A', 'churn_rate'] - ci[0][0],
        results.loc['B', 'churn_rate'] - ci[1][0]
    ]
    ax.errorbar(x=[0, 1], y=results['churn_rate'], yerr=y_err, fmt='none', c='black', capsize=5)
    
    plt.title('Comparaison des Taux de Churn (A/B Test Simulation)', fontsize=15)
    plt.ylabel('Taux de Churn (%)', fontsize=12)
    plt.ylim(0, max(results['churn_rate']) * 1.3)
    
    # Annotations
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{p.get_height():.2%}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=11, color='black', xytext=(0, 15),
                    textcoords='offset points')

    plt.savefig('ab_test_results.png')
    print("Graphique sauvegardé sous 'ab_test_results.png'.")

def interpret_and_impact(df, results, p_val):
    """Interprétation statistique et estimation de l'impact financier."""
    
    churn_rate_a = results.loc['A', 'churn_rate']
    churn_rate_b = results.loc['B', 'churn_rate']
    lift = (churn_rate_a - churn_rate_b) / churn_rate_a
    
    # Calcul de l'impact financier
    # Clients sauvés dans le groupe B par rapport au taux du groupe A
    expected_churn_b = results.loc['B', 'total'] * churn_rate_a
    actual_churn_b = results.loc['B', 'churns']
    customers_saved = max(0, expected_churn_b - actual_churn_b)
    
    # Salaire moyen des clients du groupe B qui n'ont pas churné
    avg_salary = df[df['Exited_Post'] == 0]['EstimatedSalary'].mean()
    estimated_impact = customers_saved * avg_salary
    
    print("\n" + "="*50)
    print("RÉSULTATS DE LA CAMPAGNE A/B TESTING")
    print("="*50)
    print(f"Taux Churn Contrôle (A) : {churn_rate_a:.2%}")
    print(f"Taux Churn Test (B)      : {churn_rate_b:.2%}")
    print(f"Réduction relative (Lift): {lift:.2%}")
    print(f"P-value (Z-test)         : {p_val:.4f}")
    
    sig_text = "SIGNIFICATIF" if p_val < 0.05 else "NON SIGNIFICATIF"
    print(f"\nConclusion Statistique : Le résultat est {sig_text} (seuil 0.05).")
    
    print(f"\nIMPACT FINANCIER ESTIMÉ :")
    print(f"- Clients sauvés (est.)  : {customers_saved:.1f}")
    print(f"- Salaire moyen clients  : {avg_salary:,.2f} €")
    print(f"- Valeur totale sauvée   : {estimated_impact:,.2f} €")
    print("="*50)
    
    # Bloc de commentaires pour le script final
    """
    Interprétation :
    - Si p-value < 0.05 : L'offre promotionnelle a un impact statistiquement robuste sur la réduction du churn.
    - Impact Financier : Basé sur le salaire moyen des clients 'sauvés', cela représente la masse salariale 
      potentielle conservée dans le portefeuille de la banque.
    """

if __name__ == "__main__":
    # Pipeline principal
    churn_df = load_and_preprocess('Churn_Modelling.csv')
    churn_df = run_ab_test_simulation(churn_df)
    results_stats, z, p, intervals = analyze_results(churn_df)
    
    visualize_results(results_stats, intervals)
    interpret_and_impact(churn_df, results_stats, p)
