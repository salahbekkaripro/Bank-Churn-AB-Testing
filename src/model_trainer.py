import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def train_churn_model(df):
    """
    Prépare les données et entraîne un modèle pour prédire le départ des clients.
    """
    # On fait une copie pour ne pas modifier les données d'origine
    df_model = df.copy()
    
    # On nettoie les noms de colonnes (enlève les espaces vides)
    df_model.columns = [c.strip() for c in df_model.columns]
    
    # On enlève les colonnes qui ne servent pas au modèle (ID, Nom, etc.)
    cols_to_drop = ['RowNumber', 'CustomerId', 'Surname']
    df_model = df_model.drop(columns=cols_to_drop)
    
    # Transformation des textes (Pays, Genre) en chiffres pour que l'algorithme comprenne
    le_geo = LabelEncoder()
    le_gender = LabelEncoder()
    
    df_model['Geography'] = le_geo.fit_transform(df_model['Geography'])
    df_model['Gender'] = le_gender.fit_transform(df_model['Gender'])
    
    # Séparation des variables explicatives et de la cible (Exited)
    X = df_model.drop('Exited', axis=1)
    y = df_model['Exited']
    feature_names = X.columns.tolist()
    
    # Configuration de l'algorithme XGBoost avec des paramètres de base
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # Entraînement sur l'ensemble des données pour la simulation
    model.fit(X, y)
    
    # On ajoute la probabilité de départ dans le tableau principal
    df['Churn_Probability'] = model.predict_proba(X)[:, 1]
    
    return model, df, feature_names
