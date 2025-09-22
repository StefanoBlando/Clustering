"""
Ensemble clustering and consensus methods extracted from notebook
"""

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering

def consensus_clustering(X, base_algorithms, n_runs=20, random_state=42):
    """
    Consensus clustering across multiple algorithms
    Estratto da MODULO ENSEMBLE METHODS
    """
    n_samples = X.shape[0]
    consensus_matrix = np.zeros((n_samples, n_samples))
    
    np.random.seed(random_state)
    
    algorithm_results = []
    
    for run in range(n_runs):
        for alg_name, algorithm in base_algorithms.items():
            try:
                # Bootstrap sample
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_boot = X[indices]
                
                # Fit algorithm
                labels_boot = algorithm.fit_predict(X_boot)
                
                # Update consensus matrix per gli indici originali
                for i in range(n_samples):
                    for j in range(i+1, n_samples):
                        if indices[i] < len(labels_boot) and indices[j] < len(labels_boot):
                            if labels_boot[indices[i]] == labels_boot[indices[j]]:
                                consensus_matrix[i, j] += 1
                                consensus_matrix[j, i] += 1
                
                algorithm_results.append({
                    'algorithm': alg_name,
                    'run': run,
                    'labels': labels_boot,
                    'success': True
                })
                
            except Exception as e:
                algorithm_results.append({
                    'algorithm': alg_name, 
                    'run': run,
                    'error': str(e),
                    'success': False
                })
    
    # Normalizzazione consensus matrix
    successful_runs = sum(1 for r in algorithm_results if r['success'])
    if successful_runs > 0:
        consensus_matrix /= successful_runs
    
    # Final clustering su consensus matrix
    distance_matrix = 1 - consensus_matrix
    
    try:
        final_clusterer = AgglomerativeClustering(
            n_clusters=3, 
            metric='precomputed',
            linkage='average'
        )
        consensus_labels = final_clusterer.fit_predict(distance_matrix)
    except:
        # Fallback se consensus clustering fallisce
        consensus_labels = np.zeros(n_samples)
    
    return {
        'consensus_labels': consensus_labels,
        'consensus_matrix': consensus_matrix,
        'algorithm_results': algorithm_results,
        'n_successful_runs': successful_runs
    }

def weighted_ensemble_clustering(X, algorithms_with_weights):
    """
    Weighted ensemble clustering
    Estratto da ensemble methods
    """
    ensemble_labels = []
    algorithm_labels = {}
    
    # Esegui tutti gli algoritmi
    for (alg_name, algorithm), weight in algorithms_with_weights.items():
        try:
            labels = algorithm.fit_predict(X)
            algorithm_labels[alg_name] = labels
        except Exception as e:
            print(f"Algoritmo {alg_name} fallito: {e}")
            continue
    
    if not algorithm_labels:
        return None
    
    n_samples = X.shape[0]
    
    # Voto pesato per ogni campione
    final_labels = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        # Raccoglie voti pesati
        votes = {}
        for alg_name, labels in algorithm_labels.items():
            if i < len(labels):
                cluster_id = labels[i]
                weight = algorithms_with_weights[(alg_name, None)]  # Simplified
                
                if cluster_id in votes:
                    votes[cluster_id] += weight
                else:
                    votes[cluster_id] = weight
        
        # Assegna cluster con voto maggiore
        if votes:
            final_labels[i] = max(votes.keys(), key=lambda k: votes[k])
    
    return {
        'ensemble_labels': final_labels,
        'algorithm_labels': algorithm_labels,
        'voting_weights': algorithms_with_weights
    }

def stability_across_parameters(X, clusterer_class, param_grid, n_bootstrap=50):
    """
    Test stabilità attraverso parametri
    Estratto da validation modules
    """
    from sklearn.utils import resample
    
    stability_results = []
    
    for params in param_grid:
        print(f"Testing parameters: {params}")
        
        # Base clustering con parametri
        base_clusterer = clusterer_class(**params)
        base_labels = base_clusterer.fit_predict(X)
        
        bootstrap_aris = []
        
        # Bootstrap test
        for i in range(n_bootstrap):
            try:
                X_boot = resample(X, random_state=i)
                boot_clusterer = clusterer_class(**params)
                boot_labels = boot_clusterer.fit_predict(X_boot)
                
                # ARI con base (usando lunghezza minima)
                min_len = min(len(base_labels), len(boot_labels))
                if min_len > 0:
                    ari = adjusted_rand_score(base_labels[:min_len], boot_labels[:min_len])
                    bootstrap_aris.append(ari)
                    
            except Exception as e:
                continue
        
        if bootstrap_aris:
            mean_ari = np.mean(bootstrap_aris)
            std_ari = np.std(bootstrap_aris)
        else:
            mean_ari = std_ari = 0.0
        
        stability_results.append({
            'parameters': params,
            'mean_bootstrap_ari': mean_ari,
            'std_bootstrap_ari': std_ari,
            'n_successful_bootstrap': len(bootstrap_aris)
        })
    
    return stability_results
