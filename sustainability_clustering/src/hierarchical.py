"""
Hierarchical clustering functions extracted from MODULO 5
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

def perform_hierarchical_clustering(X, method='ward', k_range=range(2, 6)):
    """
    Clustering gerarchico con diversi K
    Estratto da MODULO 5 STEP 4
    """
    hier_results = []
    
    for k in k_range:
        clusterer = AgglomerativeClustering(n_clusters=k, linkage=method)
        labels = clusterer.fit_predict(X)
        
        n_unique = len(np.unique(labels))
        if n_unique == k and n_unique > 1:
            sil = silhouette_score(X, labels)
            ch = calinski_harabasz_score(X, labels)
            
            hier_results.append({
                'k': k,
                'silhouette': sil,
                'calinski_harabasz': ch,
                'linkage': method,
                'labels': labels
            })
    
    return hier_results

def find_optimal_k_hierarchical(hier_results):
    """
    Selezione K ottimale per clustering gerarchico
    Estratto da MODULO 5 STEP 4
    """
    if not hier_results:
        return None, None
    
    # Criterio silhouette
    best_idx = np.argmax([r['silhouette'] for r in hier_results])
    optimal_k = hier_results[best_idx]['k']
    best_silhouette = hier_results[best_idx]['silhouette']
    
    return optimal_k, best_silhouette

def create_linkage_matrix(X, method='ward'):
    """
    Calcola linkage matrix per dendrogramma
    Estratto da MODULO 5 STEP 2
    """
    linkage_matrix = linkage(X, method=method, metric='euclidean')
    return linkage_matrix

def plot_dendrogram(linkage_matrix, labels=None, figsize=(20, 10)):
    """
    Plot dendrogramma completo
    Estratto da MODULO 5 STEP 3
    """
    plt.figure(figsize=figsize)
    dendr = dendrogram(linkage_matrix, 
                       labels=labels,
                       leaf_rotation=90,
                       leaf_font_size=6,
                       color_threshold=0.7*linkage_matrix[:, 2].max())
    
    plt.title('Dendrogramma Completo - Ward Linkage', fontsize=16)
    plt.xlabel('Osservazioni', fontsize=12)
    plt.ylabel('Distanza Ward', fontsize=12)
    
    return dendr

def bootstrap_stability_analysis(X, clusterer, n_bootstrap=100):
    """
    Bootstrap stability analysis
    Estratto da MODULO BOOTSTRAP VALIDAZIONE
    """
    from sklearn.utils import resample
    from sklearn.metrics import adjusted_rand_score
    
    # Labels originali
    original_labels = clusterer.fit_predict(X)
    
    bootstrap_aris = []
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        X_boot = resample(X, random_state=i)
        
        # Clustering su bootstrap
        labels_boot = clusterer.fit_predict(X_boot)
        
        # ARI con originale
        ari = adjusted_rand_score(original_labels, labels_boot)
        bootstrap_aris.append(ari)
    
    return bootstrap_aris, np.mean(bootstrap_aris), np.std(bootstrap_aris)
