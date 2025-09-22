"""
Data loading and preprocessing functions extracted from MODULO 1
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_sustainability_dataset(filepath):
    """
    Carica il dataset sostenibilità originale
    Estratto da MODULO 1 STEP 1
    """
    df_original = pd.read_excel(filepath, sheet_name='Questionario Sostenibilità')
    return df_original

def preprocess_demographics(df):
    """
    Preprocessing variabili demografiche
    Estratto da MODULO 1 STEP 2
    """
    df_processed = df.copy()
    
    # ID
    df_processed['id'] = range(1, len(df_processed) + 1)
    
    # Età normalizzata (q1)
    if 'q1' in df_processed.columns:
        df_processed['eta'] = (df_processed['q1'] - 18) / (70 - 18)
    
    # Genere dummy (q2)
    if 'q2' in df_processed.columns:
        df_processed['genere_donna'] = (df_processed['q2'] == 'F').astype(int)
    
    # Titolo di studio (q3)
    if 'q3' in df_processed.columns:
        education_mapping = {
            'Licenza elementare': 1,
            'Licenza media': 2,
            'Diploma': 3,
            'Laurea triennale': 4,
            'Laurea magistrale': 5,
            'Dottorato/Master': 6
        }
        df_processed['titolo_studio_ord'] = df_processed['q3'].map(education_mapping)
        df_processed['titolo_magistrale'] = (df_processed['titolo_studio_ord'] >= 5).astype(int)
    
    # Occupazione (q4)
    if 'q4' in df_processed.columns:
        df_processed['occup_studente'] = (df_processed['q4'] == 'Studente').astype(int)
        df_processed['occup_dipendente'] = df_processed['q4'].isin(['Dipendente privato', 'Dipendente pubblico']).astype(int)
    
    # Geografia (q5)
    if 'q5' in df_processed.columns:
        df_processed['geo_nord'] = (df_processed['q5'] == 'Nord').astype(int)
        df_processed['geo_centro'] = (df_processed['q5'] == 'Centro').astype(int)
        df_processed['geo_sud'] = (df_processed['q5'] == 'Sud').astype(int)
        df_processed['geo_isole'] = (df_processed['q5'] == 'Isole').astype(int)
    
    # Reddito (q6)
    if 'q6' in df_processed.columns:
        income_mapping = {
            'Meno di 15.000€': 1,
            'Tra 15.000€ e 30.000€': 2, 
            'Tra 30.000€ e 50.000€': 3,
            'Tra 50.000€ e 75.000€': 4,
            'Più di 75.000€': 5,
            'Preferisco non rispondere': np.nan
        }
        df_processed['reddito_ord'] = df_processed['q6'].map(income_mapping)
        df_processed['reddito_medio_alto'] = (df_processed['reddito_ord'] >= 3).astype(int)
    
    return df_processed

def preprocess_likert_variables(df):
    """
    Preprocessing variabili Likert
    Estratto da MODULO 1 STEP 3
    """
    likert_vars = [col for col in df.columns if col.startswith('q') and col[1:].isdigit()]
    likert_vars = [col for col in likert_vars if col not in ['q1', 'q2', 'q3', 'q4', 'q5', 'q6']]
    
    df_processed = df.copy()
    
    for var in likert_vars:
        if var in df_processed.columns:
            # Versione centrata
            df_processed[f'lik7c_{var}'] = df_processed[var] - 4
            # Versione 0-1
            df_processed[f'lik01_{var}'] = (df_processed[var] - 1) / 6
    
    return df_processed, likert_vars

def create_clustering_dataset(df, clustering_vars):
    """
    Crea dataset pulito per clustering
    Estratto da MODULO 1 STEP 4
    """
    df_clustering = df[clustering_vars].copy()
    df_clustering_clean = df_clustering.dropna()
    
    # Rimuovi variabili con varianza bassa
    variability = df_clustering_clean.var()
    low_variance_vars = variability[variability < 0.001].index.tolist()
    if low_variance_vars:
        df_clustering_clean = df_clustering_clean.drop(columns=low_variance_vars)
    
    return df_clustering_clean
