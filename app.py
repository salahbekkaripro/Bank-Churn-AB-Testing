import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
from datetime import datetime

# Config de la page
st.set_page_config(
    page_title="Dashboard Churn",
    page_icon="📊",
    layout="wide"
)

# --- CHARGEMENT DES DONNÉES ---
REPORT_PATH = 'reports/ab_test_summary.json'
DATA_PATH = 'data/Churn_Modelling.csv'

@st.cache_data
def load_report():
    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH, 'r') as f:
            return json.load(f)
    return None

@st.cache_data
def load_dataset():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df.columns = [c.strip() for c in df.columns]
        return df
    return None

report = load_report()
df = load_dataset()

# --- BARRE LATERALE ---
with st.sidebar:
    st.title("Paramètres")
    st.info(f"Version du projet : 1.0")
    st.info(f"Date : {datetime.now().strftime('%d/%m/%Y')}")
    st.divider()
    st.markdown("### Simulation d'impact")
    # Curseur pour changer le taux de clients sauvés
    simulation_rate = st.slider(
        "Taux de succès de l'offre (%)", 
        min_value=5, 
        max_value=50, 
        value=15, 
        step=5
    )

# --- TITRE PRINCIPAL ---
st.title("Résultats de l'A/B Test : Rétention des clients")
st.markdown("""
Ce dashboard montre les résultats de notre simulation de campagne de rétention pour réduire les départs de clients.
""")

if report:
    # --- AFFICHAGE DES CHIFFRES CLES ---
    st.divider()
    col1, col2, col3 = st.columns(3)

    # 1. Baisse du Churn
    lift_val = report['metrics']['lift']
    col1.metric("Réduction du Churn (Lift)", f"{lift_val:.2%}")

    # 2. Test Statistique
    p_val = report['statistics']['p_value']
    is_sig = report['statistics']['is_significant']
    sig_label = "Résultat Significatif" if is_sig else "Résultat Non Significatif"
    col2.metric("P-Value (Z-test)", f"{p_val:.4f}")
    if is_sig:
        col2.success(sig_label)
    else:
        col2.warning(sig_label)

    # 3. Estimation financière
    base_impact = report['business_impact']['estimated_financial_gain_euro']
    # Calcul simple pour la simulation dynamique
    dynamic_impact = (base_impact / 0.15) * (simulation_rate / 100)
    
    col3.metric(
        f"Gain estimé ({simulation_rate}%)", 
        f"{dynamic_impact:,.0f} €"
    )

    # --- GRAPHIQUES ---
    st.divider()
    st.subheader("Analyse par pays")
    
    if df is not None:
        c1, c2 = st.columns([2, 1])
        
        # Graphique des départs par pays
        with c1:
            geo_churn = df.groupby(['Geography', 'Exited']).size().reset_index(name='Count')
            geo_churn['Etat'] = geo_churn['Exited'].map({0: 'Resté', 1: 'Parti'})
            
            fig = px.bar(
                geo_churn, 
                x="Geography", 
                y="Count", 
                color="Etat",
                title="Nombre de clients par pays",
                barmode="group",
                color_discrete_map={'Resté': '#2ecc71', 'Parti': '#e74c3c'},
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Petit tableau récapitulatif
        with c2:
            st.markdown("#### Salaires Moyens")
            avg_sal = df.groupby('Geography')['EstimatedSalary'].mean().sort_values(ascending=False)
            st.dataframe(avg_sal.map("{:,.2f} €".format), use_container_width=True)
            
            st.markdown("#### Taux de départ réel")
            churn_rates = df.groupby('Geography')['Exited'].mean().sort_values(ascending=False)
            st.dataframe(churn_rates.map("{:.2%}".format), use_container_width=True)

    st.divider()
    st.caption("Projet de Licence 3 - Analyse de données")

else:
    st.error("Le fichier de résultats est introuvable. Lancez 'python main.py' d'abord.")
