"""
Factor Analysis clustering functions extracted from MODULO 8
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

def test_factorability(df_likert):
    """
    Test di fattorabilità
    Estratto da MODULO 8 STEP 2
    """
    # Test di Bartlett
    chi_square_value, p_value = calculate_bartlett_sphericity(df_likert)
    
    # Kaiser-Meyer-Olkin test
    kmo_all, kmo_model = calculate_kmo(df_likert)
    
    factorability = {
        'bartlett_chi2': chi_square_value,
        'bartlett_p': p_value,
        'kmo_model': kmo_model,
        'is_factorable': p_value < 0.05 and kmo_model > 0.6
    }
    
    return factorability

def determine_n_factors(df_likert, max_factors=8):
    """
    Determinazione numero fattori ottimale
    Estratto da MODULO 8 STEP 3
    """
    from sklearn.decomposition import PCA
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_likert)
    
    # Kaiser criterion
    pca = PCA()
    pca.fit(X_scaled)
    eigenvalues = pca.explained_variance_
    kaiser_factors = np.sum(eigenvalues > 1)
    
    # Varianza spiegata
    cum_var_explained = np.cumsum(pca.explained_variance_ratio_)
    factors_70pct = np.argmax(cum_var_explained >= 0.70) + 1
    
    return {
        'kaiser_factors': kaiser_factors,
        'factors_70pct': factors_70pct,
        'eigenvalues': eigenvalues,
        'cum_var_explained': cum_var_explained
    }

def perform_factor_analysis(df_likert, n_factors):
    """
    Analisi fattoriale finale
    Estratto da MODULO 8 STEP 5
    """
    fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax', method='minres')
    fa.fit(df_likert)
    
    # Risultati
    loadings = fa.loadings_
    communalities = fa.get_communalities()
    eigenvals = fa.get_eigenvalues()[0]
    
    # Factor scores
    factor_scores = fa.transform(df_likert)
    
    return {
        'factor_analyzer': fa,
        'loadings': loadings,
        'communalities': communalities,
        'eigenvals': eigenvals,
        'factor_scores': factor_scores,
        'variance_explained': np.sum(eigenvals) / len(df_likert.columns)
    }

def cluster_on_factors(factor_scores, k_range=range(2, 7)):
    """
    Clustering sui factor scores
    Estratto da MODULO 8 STEP 7
    """
    scaler = StandardScaler()
    factor_scores_scaled = scaler.fit_transform(factor_scores)
    
    clustering_results = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(factor_scores_scaled)
        
        n_unique = len(np.unique(labels))
        if n_unique == k:
            sil = silhouette_score(factor_scores_scaled, labels)
            
            clustering_results.append({
                'k': k,
                'silhouette': sil,
                'labels': labels,
                'inertia': kmeans.inertia_
            })
    
    return clustering_results
