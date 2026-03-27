import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Churn Retention Dashboard",
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

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Configuration")
    st.info(f"**Version Modèle :** v1.2.0")
    st.info(f"**Dernier Test :** {datetime.now().strftime('%d/%m/%Y')}")
    st.divider()
    st.markdown("### 🛠️ Paramètres Simulation")
    # Curseur pour simuler l'impact financier dynamique
    simulation_rate = st.slider(
        "Taux de clients sauvés (%)", 
        min_value=5, 
        max_value=50, 
        value=15, 
        step=5
    )

# --- HEADER ---
st.title("📊 Analyse d'A/B Testing : Campagne de Rétention Churn")
st.markdown("""
Cette interface présente les résultats de la dernière campagne d'A/B Testing visant à réduire le churn bancaire via une offre promotionnelle ciblée.
""")

if report:
    # --- SECTION KPI ---
    st.divider()
    col1, col2, col3 = st.columns(3)

    # 1. Lift
    lift_val = report['metrics']['lift']
    col1.metric("Lift (Réduction Churn)", f"{lift_val:.2%}", delta=f"{lift_val*100:.1f}%")

    # 2. P-Value & Significativité
    p_val = report['statistics']['p_value']
    is_sig = report['statistics']['is_significant']
    sig_label = "✅ SIGNIFICATIF" if is_sig else "❌ NON SIGNIFICATIF"
    col2.metric("P-Value (Z-test)", f"{p_val:.4f}", help="Seuil de significativité < 0.05")
    if is_sig:
        col2.success(sig_label)
    else:
        col2.error(sig_label)

    # 3. Impact Financier Statique vs Dynamique
    base_impact = report['business_impact']['estimated_financial_gain_euro']
    # Calcul dynamique basé sur le slider
    # On recalcule l'impact : (Base Saved / 0.15) * (Simulation Rate / 100) * AvgSalary
    # Note: On simplifie ici en utilisant une règle de trois sur l'impact de base
    dynamic_impact = (base_impact / 0.15) * (simulation_rate / 100)
    
    col3.metric(
        f"Impact Financier ({simulation_rate}%)", 
        f"{dynamic_impact:,.0f} €", 
        delta=f"{(dynamic_impact - base_impact):+,.0f} € vs base"
    )

    # --- SECTION VISUALISATION ---
    st.divider()
    st.subheader("🌍 Analyse du Churn par Géographie (Données Réelles)")
    
    if df is not None:
        c1, c2 = st.columns([2, 1])
        
        # Graphique Plotly
        with c1:
            geo_churn = df.groupby(['Geography', 'Exited']).size().reset_index(name='Count')
            geo_churn['Exited'] = geo_churn['Exited'].map({0: 'Resté', 1: 'Parti'})
            
            fig = px.bar(
                geo_churn, 
                x="Geography", 
                y="Count", 
                color="Exited",
                title="Répartition Churn par Pays",
                barmode="group",
                color_discrete_map={'Resté': '#2ecc71', 'Parti': '#e74c3c'},
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Tableau de données
        with c2:
            st.markdown("#### Top Salaires Moyens")
            avg_sal = df.groupby('Geography')['EstimatedSalary'].mean().sort_values(ascending=False)
            st.dataframe(avg_sal.map("{:,.2f} €".format), use_container_width=True)
            
            st.markdown("#### Taux de Churn Réel")
            churn_rates = df.groupby('Geography')['Exited'].mean().sort_values(ascending=False)
            st.dataframe(churn_rates.map("{:.2%}".format), use_container_width=True)

    # --- FOOTER ---
    st.divider()
    st.caption("Généré par Gemini CLI - Senior Data Science Solution")

else:
    st.warning("⚠️ Rapport JSON introuvable. Veuillez exécuter 'python main.py' pour générer les résultats.")
