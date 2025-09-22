"""
Feature engineering functions extracted from various modules
"""

import pandas as pd
import numpy as np

def prepare_mca_data(df):
    """
    Preparazione dati per MCA
    Estratto da MODULO 6 STEP 1
    """
    categorical_vars_mca = ['q2', 'q3', 'q4', 'q5', 'q6']
    
    df_mca = df[categorical_vars_mca].copy()
    
    # Encoding per MCA
    for var in categorical_vars_mca:
        if var in df_mca.columns:
            df_mca[var] = df_mca[var].astype(str)
    
    return df_mca, categorical_vars_mca

def prepare_factor_data(df):
    """
    Preparazione dataset per Factor Analysis
    Estratto da MODULO 8 STEP 1
    """
    likert_vars = ['q7', 'q8', 'q9', 'q10', 'q11', 'q12', 'q13', 'q14', 
                   'q16', 'q18', 'q19', 'q20', 'q21', 'q23', 'q24', 'q29']
    
    existing_likert = [var for var in likert_vars if var in df.columns]
    df_likert = df[existing_likert].copy()
    df_likert = df_likert.fillna(df_likert.mean())
    
    return df_likert, existing_likert

def prepare_fuzzy_data(df):
    """
    Preparazione dati per Fuzzy C-Means
    Estratto da moduli Fuzzy
    """
    df_encoded = df.copy()
    
    df_encoded['eta_norm'] = (df_encoded['q1'] - 18) / (70 - 18)
    df_encoded['genere_donna'] = (df_encoded['q2'] == 'Donna').astype(int)
    df_encoded['titolo_magistrale'] = (df_encoded['q3'] == 'Laurea Magistrale').astype(int)
    df_encoded['occup_studente'] = (df_encoded['q4'] == 'Studente/ Studentessa').astype(int)
    df_encoded['geo_centro'] = (df_encoded['q5'] == 'Centro').astype(int)
    df_encoded['reddito_alto'] = df_encoded['q6'].isin(['30001 - 50000', 'Più di 50000']).astype(int)
    
    # Likert normalization
    likert_vars = ['q8', 'q19', 'q20', 'q21']
    for var in likert_vars:
        if var in df_encoded.columns:
            df_encoded[f'{var}_norm'] = (df_encoded[var] - 1) / 6

    fuzzy_vars = ['eta_norm', 'genere_donna', 'titolo_magistrale', 'occup_studente', 
                  'geo_centro', 'reddito_alto'] + [f'{var}_norm' for var in likert_vars if var in df.columns]
    
    return df_encoded, fuzzy_vars
