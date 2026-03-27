import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def train_churn_model(df):
    """
    Prépare les données et entraîne un modèle XGBoost pour prédire le churn.
    
    Args:
        df (pd.DataFrame): Dataset original.
        
    Returns:
        tuple: (model, df_with_probs, feature_names)
    """
    # 1. Préparation des données
    # On travaille sur une copie pour ne pas altérer le df original
    df_model = df.copy()
    
    # Nettoyage (stripping) au cas où ce n'est pas déjà fait
    df_model.columns = [c.strip() for c in df_model.columns]
    
    # Drop colonnes non pertinentes pour le ML
    cols_to_drop = ['RowNumber', 'CustomerId', 'Surname']
    df_model = df_model.drop(columns=cols_to_drop)
    
    # 2. Encodage des variables catégorielles
    le_geo = LabelEncoder()
    le_gender = LabelEncoder()
    
    df_model['Geography'] = le_geo.fit_transform(df_model['Geography'])
    df_model['Gender'] = le_gender.fit_transform(df_model['Gender'])
    
    # 3. Séparation Features / Target
    X = df_model.drop('Exited', axis=1)
    y = df_model['Exited']
    feature_names = X.columns.tolist()
    
    # 4. Entraînement du modèle
    # On utilise des hyperparamètres raisonnables par défaut
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # On entraîne sur tout le dataset pour la simulation, 
    # mais en situation réelle on ferait un split train/test.
    model.fit(X, y)
    
    # 5. Calcul des probabilités de churn pour chaque client
    df['Churn_Probability'] = model.predict_proba(X)[:, 1]
    
    return model, df, feature_names
