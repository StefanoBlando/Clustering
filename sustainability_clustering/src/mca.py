"""
Multiple Correspondence Analysis clustering functions extracted from MODULO 6
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def perform_mca_preprocessing(df, categorical_vars):
    """
    Preprocessing per MCA
    Estratto da MODULO 6 STEP 2
    """
    # Dummy encoding per MCA
    df_dummies = pd.get_dummies(df[categorical_vars], drop_first=False, dummy_na=True)
    
    return df_dummies

def calculate_mca_coordinates(df_dummies):
    """
    Calcolo coordinate MCA (implementazione semplificata)
    Estratto da MODULO 6 STEP 3
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Standardizzazione
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_dummies)
    
    # PCA come approssimazione MCA
    pca = PCA(n_components=min(5, X_scaled.shape[1]))
    coordinates = pca.fit_transform(X_scaled)
    
    # Informazioni dimensioni
    explained_variance = pca.explained_variance_ratio_
    
    return coordinates, explained_variance, pca

def mca_clustering(coordinates, k_range=range(2, 6)):
    """
    Clustering su coordinate MCA
    Estratto da MODULO 6 STEP 4
    """
    mca_results = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coordinates)
        
        n_unique = len(np.unique(labels))
        if n_unique == k:
            sil = silhouette_score(coordinates, labels)
            
            mca_results.append({
                'k': k,
                'silhouette': sil,
                'labels': labels,
                'kmeans_model': kmeans
            })
    
    return mca_results

def find_optimal_mca_k(mca_results):
    """
    Selezione K ottimale per MCA
    Estratto da MODULO 6 STEP 4
    """
    if not mca_results:
        return None, None
    
    best_idx = np.argmax([r['silhouette'] for r in mca_results])
    optimal_k = mca_results[best_idx]['k']
    best_silhouette = mca_results[best_idx]['silhouette']
    
    return optimal_k, best_silhouette
