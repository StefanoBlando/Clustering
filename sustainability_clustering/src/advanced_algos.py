"""
Advanced clustering algorithms extracted from MODULO CROSS-DATASET
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.neighbors import kneighbors_graph
import warnings
warnings.filterwarnings('ignore')

def dbscan_parameter_optimization(X, eps_range=None, min_samples_range=None):
    """
    DBSCAN parameter optimization
    Estratto da MODULO CROSS-DATASET DBSCAN
    """
    if eps_range is None:
        eps_range = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    if min_samples_range is None:
        min_samples_range = [3, 5, 8, 10, 15]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    dbscan_results = []
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            try:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(X_scaled)
                
                # Analisi risultati
                unique_labels = np.unique(labels)
                n_clusters = len(unique_labels) - (1 if -1 in labels else 0)  # Esclude noise
                n_noise = np.sum(labels == -1)
                noise_ratio = n_noise / len(labels)
                
                if n_clusters >= 2:
                    # Calcola silhouette escludendo noise
                    mask = labels != -1
                    if mask.sum() > n_clusters:
                        silhouette = silhouette_score(X_scaled[mask], labels[mask])
                    else:
                        silhouette = 0.0
                else:
                    silhouette = 0.0
                
                dbscan_results.append({
                    'eps': eps,
                    'min_samples': min_samples,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'noise_ratio': noise_ratio,
                    'silhouette': silhouette,
                    'labels': labels
                })
                
            except Exception as e:
                continue
    
    return dbscan_results

def spectral_clustering_comprehensive(X, affinity_types=['rbf', 'nearest_neighbors'], k_range=range(2, 6)):
    """
    Spectral clustering con diversi kernel
    Estratto da MODULO CROSS-DATASET SPECTRAL
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    spectral_results = []
    
    for affinity in affinity_types:
        for k in k_range:
            if affinity == 'rbf':
                # Test diversi gamma
                gamma_values = [0.1, 1.0, 'scale', 'auto']
                for gamma in gamma_values:
                    try:
                        spectral = SpectralClustering(
                            n_clusters=k, 
                            affinity=affinity,
                            gamma=gamma,
                            random_state=42
                        )
                        labels = spectral.fit_predict(X_scaled)
                        
                        n_unique = len(np.unique(labels))
                        if n_unique == k:
                            silhouette = silhouette_score(X_scaled, labels)
                            calinski_h = calinski_harabasz_score(X_scaled, labels)
                            
                            spectral_results.append({
                                'affinity': affinity,
                                'gamma': gamma,
                                'k': k,
                                'silhouette': silhouette,
                                'calinski_harabasz': calinski_h,
                                'labels': labels,
                                'n_clusters_found': n_unique
                            })
                            
                    except Exception as e:
                        continue
                        
            elif affinity == 'nearest_neighbors':
                # Test diversi n_neighbors
                n_neighbors_values = [5, 10, 15]
                for n_neighbors in n_neighbors_values:
                    try:
                        spectral = SpectralClustering(
                            n_clusters=k,
                            affinity=affinity,
                            n_neighbors=n_neighbors,
                            random_state=42
                        )
                        labels = spectral.fit_predict(X_scaled)
                        
                        n_unique = len(np.unique(labels))
                        if n_unique == k:
                            silhouette = silhouette_score(X_scaled, labels)
                            calinski_h = calinski_harabasz_score(X_scaled, labels)
                            
                            spectral_results.append({
                                'affinity': affinity,
                                'n_neighbors': n_neighbors,
                                'k': k,
                                'silhouette': silhouette,
                                'calinski_harabasz': calinski_h,
                                'labels': labels,
                                'n_clusters_found': n_unique
                            })
                            
                    except Exception as e:
                        continue
    
    return spectral_results

def analyze_dbscan_patterns(dbscan_results):
    """
    Analisi pattern DBSCAN
    Estratto da moduli DBSCAN analysis
    """
    if not dbscan_results:
        return {'error': 'No valid DBSCAN results'}
    
    df_results = pd.DataFrame(dbscan_results)
    
    analysis = {
        'total_configurations': len(df_results),
        'successful_clustering': len(df_results[df_results['n_clusters'] >= 2]),
        'mean_noise_ratio': df_results['noise_ratio'].mean(),
        'noise_range': (df_results['noise_ratio'].min(), df_results['noise_ratio'].max()),
        'best_configuration': None,
        'clustering_feasibility': 'Low' if df_results['n_clusters'].max() <= 1 else 'Medium'
    }
    
    # Migliore configurazione
    valid_results = df_results[df_results['n_clusters'] >= 2]
    if len(valid_results) > 0:
        best_idx = valid_results['silhouette'].idxmax()
        analysis['best_configuration'] = valid_results.loc[best_idx].to_dict()
        
        if analysis['best_configuration']['silhouette'] > 0.3:
            analysis['clustering_feasibility'] = 'High'
        elif analysis['best_configuration']['silhouette'] > 0.2:
            analysis['clustering_feasibility'] = 'Medium'
    
    return analysis

def find_optimal_spectral_config(spectral_results):
    """
    Trova configurazione Spectral ottimale
    Estratto da spectral analysis
    """
    if not spectral_results:
        return None
    
    df_spectral = pd.DataFrame(spectral_results)
    
    # Configurazione con migliore silhouette
    best_idx = df_spectral['silhouette'].idxmax()
    optimal_config = df_spectral.loc[best_idx].to_dict()
    
    # Statistiche per affinity type
    affinity_stats = df_spectral.groupby('affinity')['silhouette'].agg(['mean', 'max', 'count'])
    
    return {
        'optimal_configuration': optimal_config,
        'affinity_comparison': affinity_stats.to_dict(),
        'performance_summary': {
            'best_silhouette': optimal_config['silhouette'],
            'best_affinity': optimal_config['affinity'],
            'best_k': optimal_config['k']
        }
    }
