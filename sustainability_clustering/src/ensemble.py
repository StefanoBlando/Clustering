"""
Ensemble clustering methods
Estratto da validation ensemble methods del notebook
"""

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

def consensus_clustering(X, base_algorithms, n_runs=20, random_state=42):
    """
    Consensus clustering across multiple algorithms
    Estratto da MODULO ENSEMBLE METHODS
    """
    n_samples = X.shape[0]
    consensus_matrix = np.zeros((n_samples, n_samples))
    
    np.random.seed(random_state)
    algorithm_results = []
    successful_runs = 0
    
    print(f"Running consensus clustering: {len(base_algorithms)} algorithms × {n_runs} runs")
    
    for run in range(n_runs):
        for alg_name, algorithm in base_algorithms.items():
            try:
                # Bootstrap sample
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_boot = X[indices]
                
                # Fit algorithm
                labels_boot = algorithm.fit_predict(X_boot)
                
                # Controlla validità clustering
                if len(np.unique(labels_boot)) > 1:
                    # Update consensus matrix
                    for i in range(n_samples):
                        for j in range(i+1, n_samples):
                            # Trova dove sono finiti i samples originali nel bootstrap
                            orig_i_positions = np.where(indices == i)[0]
                            orig_j_positions = np.where(indices == j)[0]
                            
                            # Se entrambi sono nel bootstrap
                            if len(orig_i_positions) > 0 and len(orig_j_positions) > 0:
                                # Prendi la prima occorrenza
                                boot_i = orig_i_positions[0]
                                boot_j = orig_j_positions[0]
                                
                                # Se stesso cluster
                                if labels_boot[boot_i] == labels_boot[boot_j]:
                                    consensus_matrix[i, j] += 1
                                    consensus_matrix[j, i] += 1
                    
                    successful_runs += 1
                
                algorithm_results.append({
                    'algorithm': alg_name,
                    'run': run,
                    'labels': labels_boot,
                    'n_clusters': len(np.unique(labels_boot)),
                    'success': True
                })
                
            except Exception as e:
                algorithm_results.append({
                    'algorithm': alg_name, 
                    'run': run,
                    'error': str(e),
                    'success': False
                })
    
    print(f"Successful runs: {successful_runs}")
    
    # Normalizza consensus matrix
    if successful_runs > 0:
        consensus_matrix = consensus_matrix / successful_runs
    
    # Final clustering su consensus matrix
    distance_matrix = 1 - consensus_matrix + np.eye(n_samples) * 1e-10  # Evita diagonale zero
    
    try:
        # Prova diversi K per trovare il migliore
        best_labels = None
        best_silhouette = -1
        
        for k in [2, 3, 4]:
            try:
                final_clusterer = AgglomerativeClustering(
                    n_clusters=k, 
                    metric='precomputed',
                    linkage='average'
                )
                test_labels = final_clusterer.fit_predict(distance_matrix)
                
                if len(np.unique(test_labels)) == k:
                    sil = silhouette_score(X, test_labels)
                    if sil > best_silhouette:
                        best_silhouette = sil
                        best_labels = test_labels
            except:
                continue
        
        if best_labels is None:
            # Fallback
            best_labels = np.zeros(n_samples, dtype=int)
            
    except Exception as e:
        print(f"Consensus clustering fallito: {e}")
        best_labels = np.zeros(n_samples, dtype=int)
    
    return {
        'consensus_labels': best_labels,
        'consensus_matrix': consensus_matrix,
        'algorithm_results': algorithm_results,
        'n_successful_runs': successful_runs,
        'final_silhouette': best_silhouette if 'best_silhouette' in locals() else 0.0
    }

def weighted_ensemble_clustering(X, algorithms_weights, voting_strategy='weighted'):
    """
    Weighted ensemble clustering
    Estratto da ensemble methods
    """
    algorithm_labels = {}
    algorithm_weights = {}
    
    # Esegui tutti gli algoritmi
    for (alg_name, algorithm), weight in algorithms_weights.items():
        try:
            labels = algorithm.fit_predict(X)
            if len(np.unique(labels)) > 1:
                algorithm_labels[alg_name] = labels
                algorithm_weights[alg_name] = weight
        except Exception as e:
            print(f"Algoritmo {alg_name} fallito: {e}")
            continue
    
    if not algorithm_labels:
        return None
    
    n_samples = X.shape[0]
    
    if voting_strategy == 'weighted':
        # Voto pesato per ogni campione
        final_labels = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            votes = {}
            total_weight = 0
            
            for alg_name, labels in algorithm_labels.items():
                if i < len(labels):
                    cluster_id = labels[i]
                    weight = algorithm_weights[alg_name]
                    
                    if cluster_id not in votes:
                        votes[cluster_id] = 0
                    votes[cluster_id] += weight
                    total_weight += weight
            
            # Assegna cluster con peso maggiore
            if votes:
                final_labels[i] = max(votes.keys(), key=lambda k: votes[k])
        
    elif voting_strategy == 'majority':
        # Voto maggioritario semplice
        final_labels = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            votes = {}
            for alg_name, labels in algorithm_labels.items():
                if i < len(labels):
                    cluster_id = labels[i]
                    votes[cluster_id] = votes.get(cluster_id, 0) + 1
            
            if votes:
                final_labels[i] = max(votes.keys(), key=lambda k: votes[k])
    
    # Metriche ensemble
    ensemble_silhouette = silhouette_score(X, final_labels) if len(np.unique(final_labels)) > 1 else 0.0
    
    # Accordo tra algoritmi
    agreement_scores = []
    alg_names = list(algorithm_labels.keys())
    
    for i in range(len(alg_names)):
        for j in range(i+1, len(alg_names)):
            ari = adjusted_rand_score(algorithm_labels[alg_names[i]], 
                                    algorithm_labels[alg_names[j]])
            agreement_scores.append(ari)
    
    mean_agreement = np.mean(agreement_scores) if agreement_scores else 0.0
    
    return {
        'ensemble_labels': final_labels,
        'ensemble_silhouette': ensemble_silhouette,
        'algorithm_labels': algorithm_labels,
        'voting_weights': algorithm_weights,
        'voting_strategy': voting_strategy,
        'mean_algorithm_agreement': mean_agreement,
        'agreement_scores': agreement_scores
    }

def stability_across_parameters(X, clusterer_class, param_grid, n_bootstrap=50, random_state=42):
    """
    Test stabilità attraverso parametri
    Estratto da validation modules
    """
    np.random.seed(random_state)
    stability_results = []
    
    print(f"Testing stability: {len(param_grid)} parameter configurations")
    
    for i, params in enumerate(param_grid):
        print(f"Configuration {i+1}/{len(param_grid)}: {params}")
        
        try:
            # Base clustering con parametri
            base_clusterer = clusterer_class(**params)
            base_labels = base_clusterer.fit_predict(X)
            
            if len(np.unique(base_labels)) <= 1:
                print("  Clustering non valido - skip")
                continue
            
            bootstrap_aris = []
            bootstrap_silhouettes = []
            
            # Bootstrap test
            for boot_iter in range(n_bootstrap):
                try:
                    X_boot = resample(X, random_state=boot_iter)
                    boot_clusterer = clusterer_class(**params)
                    boot_labels = boot_clusterer.fit_predict(X_boot)
                    
                    # ARI con base (usando lunghezza minima)
                    min_len = min(len(base_labels), len(boot_labels))
                    if min_len > 0 and len(np.unique(boot_labels)) > 1:
                        ari = adjusted_rand_score(base_labels[:min_len], boot_labels[:min_len])
                        sil = silhouette_score(X_boot, boot_labels)
                        bootstrap_aris.append(ari)
                        bootstrap_silhouettes.append(sil)
                        
                except Exception as e:
                    continue
            
            if bootstrap_aris:
                mean_ari = np.mean(bootstrap_aris)
                std_ari = np.std(bootstrap_aris)
                mean_sil = np.mean(bootstrap_silhouettes)
                
                print(f"  Bootstrap ARI: {mean_ari:.3f} ± {std_ari:.3f}")
                
                stability_results.append({
                    'parameters': params,
                    'mean_bootstrap_ari': mean_ari,
                    'std_bootstrap_ari': std_ari,
                    'mean_bootstrap_silhouette': mean_sil,
                    'n_successful_bootstrap': len(bootstrap_aris),
                    'stability_score': mean_ari * (1 - std_ari),  # Penalizza alta varianza
                    'base_silhouette': silhouette_score(X, base_labels)
                })
        except Exception as e:
            print(f"  Errore: {e}")
            continue
    
    return stability_results

def ensemble_performance_summary(ensemble_result, individual_results):
    """
    Riassunto performance ensemble vs algoritmi individuali
    """
    if not ensemble_result or not individual_results:
        return {}
    
    ensemble_sil = ensemble_result.get('ensemble_silhouette', 0)
    individual_sils = [result.get('silhouette', 0) for result in individual_results.values()]
    
    summary = {
        'ensemble_silhouette': ensemble_sil,
        'individual_silhouettes': individual_sils,
        'mean_individual': np.mean(individual_sils),
        'best_individual': np.max(individual_sils),
        'ensemble_improvement': ensemble_sil - np.mean(individual_sils),
        'beats_best_individual': ensemble_sil > np.max(individual_sils),
        'mean_algorithm_agreement': ensemble_result.get('mean_algorithm_agreement', 0),
        'n_algorithms_used': len(individual_results)
    }
    
    return summary
