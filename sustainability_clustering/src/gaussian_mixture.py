"""
Gaussian Mixture Models and Bayesian GMM
Estratto da moduli probabilistic methods del notebook
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

def gaussian_mixture_comprehensive(X, n_components_range=range(2, 8), covariance_types=['full', 'tied', 'diag', 'spherical']):
    """
    GMM completo con diverse covariance structures
    Estratto da moduli probabilistic methods
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    gmm_results = []
    
    for covariance_type in covariance_types:
        for n_components in n_components_range:
            try:
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type=covariance_type,
                    random_state=42,
                    max_iter=200,
                    n_init=3
                )
                
                gmm.fit(X_scaled)
                labels = gmm.predict(X_scaled)
                
                # Verifica convergenza e validità
                n_unique = len(np.unique(labels))
                if n_unique > 1 and gmm.converged_:
                    # Metriche performance
                    silhouette = silhouette_score(X_scaled, labels)
                    calinski_h = calinski_harabasz_score(X_scaled, labels)
                    
                    # Metriche model selection
                    log_likelihood = gmm.score(X_scaled)
                    bic = gmm.bic(X_scaled)
                    aic = gmm.aic(X_scaled)
                    
                    # Analisi componenti
                    component_weights = gmm.weights_
                    min_component_weight = np.min(component_weights)
                    
                    gmm_results.append({
                        'n_components': n_components,
                        'covariance_type': covariance_type,
                        'silhouette': silhouette,
                        'calinski_harabasz': calinski_h,
                        'log_likelihood': log_likelihood,
                        'bic': bic,
                        'aic': aic,
                        'labels': labels,
                        'gmm_model': gmm,
                        'converged': gmm.converged_,
                        'component_weights': component_weights,
                        'min_component_weight': min_component_weight,
                        'effective_components': n_unique
                    })
                    
            except Exception as e:
                continue
    
    return gmm_results

def bayesian_gmm_analysis(X, max_components=10, weight_concentration_prior=1.0):
    """
    Bayesian GMM con automatic relevance determination
    Estratto da BGMM implementation nel notebook
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    bgmm_results = []
    
    # Test diverse configurazioni
    covariance_types = ['diag', 'full']  # Spherical e tied spesso problematici
    weight_concentrations = [weight_concentration_prior, weight_concentration_prior/2, weight_concentration_prior*2]
    
    for covariance_type in covariance_types:
        for weight_conc in weight_concentrations:
            for n_components in range(3, max_components + 1):
                try:
                    bgmm = BayesianGaussianMixture(
                        n_components=n_components,
                        covariance_type=covariance_type,
                        weight_concentration_prior_type='dirichlet_process',
                        weight_concentration_prior=weight_conc,
                        random_state=42,
                        max_iter=200,
                        n_init=3
                    )
                    
                    bgmm.fit(X_scaled)
                    labels = bgmm.predict(X_scaled)
                    
                    # Componenti attive (automatic relevance determination)
                    active_threshold = 0.01
                    active_components = np.sum(bgmm.weights_ > active_threshold)
                    
                    # Validità risultato
                    n_unique = len(np.unique(labels))
                    if n_unique > 1 and bgmm.converged_ and active_components >= 2:
                        
                        # Metriche performance
                        silhouette = silhouette_score(X_scaled, labels)
                        log_likelihood = bgmm.score(X_scaled)
                        
                        # Analisi weights per sparsity
                        weight_entropy = -np.sum(bgmm.weights_ * np.log(bgmm.weights_ + 1e-10))
                        weight_sparsity = np.sum(bgmm.weights_ < active_threshold) / len(bgmm.weights_)
                        
                        bgmm_results.append({
                            'n_components_theoretical': n_components,
                            'n_components_active': active_components,
                            'covariance_type': covariance_type,
                            'weight_concentration_prior': weight_conc,
                            'silhouette': silhouette,
                            'log_likelihood': log_likelihood,
                            'labels': labels,
                            'weights': bgmm.weights_,
                            'bgmm_model': bgmm,
                            'converged': bgmm.converged_,
                            'weight_entropy': weight_entropy,
                            'weight_sparsity': weight_sparsity,
                            'effective_components': n_unique
                        })
                        
                except Exception as e:
                    continue
    
    return bgmm_results

def find_optimal_gmm(gmm_results, criterion='bic'):
    """
    Trova GMM ottimale secondo criterio specificato
    Estratto da model selection nel notebook
    """
    if not gmm_results:
        return None
    
    df_gmm = pd.DataFrame([
        {k: v for k, v in result.items() if k != 'gmm_model' and k != 'labels'} 
        for result in gmm_results
    ])
    
    if criterion == 'bic':
        # BIC: lower is better
        optimal_idx = df_gmm['bic'].idxmin()
    elif criterion == 'aic':
        # AIC: lower is better  
        optimal_idx = df_gmm['aic'].idxmin()
    elif criterion == 'silhouette':
        # Silhouette: higher is better
        optimal_idx = df_gmm['silhouette'].idxmax()
    else:
        # Default: BIC
        optimal_idx = df_gmm['bic'].idxmin()
    
    optimal_config = gmm_results[optimal_idx]
    
    return {
        'optimal_configuration': optimal_config,
        'selection_criterion': criterion,
        'comparison_stats': {
            'mean_silhouette': df_gmm['silhouette'].mean(),
            'best_silhouette': df_gmm['silhouette'].max(),
            'mean_bic': df_gmm['bic'].mean(),
            'best_bic': df_gmm['bic'].min(),
            'n_configurations_tested': len(df_gmm)
        }
    }

def analyze_component_separation(gmm_model, X):
    """
    Analizza separazione tra componenti GMM
    Estratto da component analysis nel notebook
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Probabilità appartenenza per ogni componente
    responsibilities = gmm_model.predict_proba(X_scaled)
    
    # Analisi overlap tra componenti
    max_responsibilities = np.max(responsibilities, axis=1)
    mean_max_resp = np.mean(max_responsibilities)
    
    # Entropia per misurare incertezza
    entropy = -np.sum(responsibilities * np.log(responsibilities + 1e-10), axis=1)
    mean_entropy = np.mean(entropy)
    max_entropy = np.log(gmm_model.n_components)  # Theoretical maximum
    normalized_entropy = mean_entropy / max_entropy
    
    # Campioni con alta incertezza
    uncertain_threshold = 0.6  # Soglia per max responsibility
    uncertain_samples = np.sum(max_responsibilities < uncertain_threshold)
    uncertainty_percentage = uncertain_samples / len(X_scaled) * 100
    
    # Distanze tra centri componenti
    centers = gmm_model.means_
    n_components = len(centers)
    inter_component_distances = []
    
    for i in range(n_components):
        for j in range(i+1, n_components):
            dist = np.linalg.norm(centers[i] - centers[j])
            inter_component_distances.append(dist)
    
    return {
        'mean_max_responsibility': mean_max_resp,
        'mean_entropy': mean_entropy,
        'normalized_entropy': normalized_entropy,
        'uncertain_samples': uncertain_samples,
        'uncertainty_percentage': uncertainty_percentage,
        'inter_component_distances': inter_component_distances,
        'mean_inter_component_distance': np.mean(inter_component_distances) if inter_component_distances else 0,
        'component_weights': gmm_model.weights_,
        'separation_quality': 'High' if mean_max_resp > 0.8 else 'Medium' if mean_max_resp > 0.6 else 'Low'
    }

def compare_covariance_types(gmm_results):
    """
    Confronta performance per tipo covariance
    Estratto da covariance comparison nel notebook
    """
    df_results = pd.DataFrame([
        {k: v for k, v in result.items() if k not in ['gmm_model', 'labels']} 
        for result in gmm_results
    ])
    
    if df_results.empty:
        return {}
    
    # Raggruppa per covariance type
    comparison = {}
    
    for cov_type in df_results['covariance_type'].unique():
        subset = df_results[df_results['covariance_type'] == cov_type]
        
        comparison[cov_type] = {
            'mean_silhouette': subset['silhouette'].mean(),
            'best_silhouette': subset['silhouette'].max(),
            'mean_bic': subset['bic'].mean(),
            'best_bic': subset['bic'].min(),
            'mean_aic': subset['aic'].mean(),
            'best_aic': subset['aic'].min(),
            'n_configurations': len(subset),
            'convergence_rate': subset['converged'].mean(),
            'mean_components_used': subset['effective_components'].mean()
        }
    
    # Ranking per performance
    cov_ranking = sorted(comparison.keys(), 
                        key=lambda x: comparison[x]['best_silhouette'], 
                        reverse=True)
    
    return {
        'covariance_comparison': comparison,
        'ranking_by_silhouette': cov_ranking,
        'recommendations': {
            'best_overall': cov_ranking[0] if cov_ranking else None,
            'most_stable': max(comparison.keys(), key=lambda x: comparison[x]['convergence_rate']) if comparison else None
        }
    }
