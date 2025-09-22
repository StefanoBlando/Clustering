"""
Gaussian Mixture Models and probabilistic clustering
Estratto da moduli GMM
"""

import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

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
                    max_iter=200
                )
                
                gmm.fit(X_scaled)
                labels = gmm.predict(X_scaled)
                
                # Metriche
                n_unique = len(np.unique(labels))
                if n_unique > 1:
                    silhouette = silhouette_score(X_scaled, labels)
                    log_likelihood = gmm.score(X_scaled)
                    bic = gmm.bic(X_scaled)
                    aic = gmm.aic(X_scaled)
                    
                    gmm_results.append({
                        'n_components': n_components,
                        'covariance_type': covariance_type,
                        'silhouette': silhouette,
                        'log_likelihood': log_likelihood,
                        'bic': bic,
                        'aic': aic,
                        'labels': labels,
                        'gmm_model': gmm,
                        'converged': gmm.converged_
                    })
                    
            except Exception as e:
                continue
    
    return gmm_results

def bayesian_gmm_analysis(X, max_components=10):
    """
    Bayesian GMM con automatic relevance determination
    Estratto da BGMM implementation
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    bgmm_results = []
    
    covariance_types = ['full', 'diag']  # Tied e spherical spesso problematici per BGMM
    
    for covariance_type in covariance_types:
        for n_components in range(2, max_components + 1):
            try:
                bgmm = BayesianGaussianMixture(
                    n_components=n_components,
                    covariance_type=covariance_type,
                    weight_concentration_prior_type='dirichlet_process',
                    random_state=42,
                    max_iter=200
                )
                
                bgmm.fit(X_scaled)
                labels = bgmm.predict(X_scaled)
                
                # Componenti attive (weight > soglia)
                active_components = np.sum(bgmm.weights_ > 0.01)
                
                # Metriche
                if len(np.unique(labels)) > 1:
                    silhouette = silhouette_score(X_scaled, labels)
                    log_likelihood = bgmm.score(X_scaled)
                    
                    bgmm_results.append({
                        'n_components_theoretical': n_components,
                        'n_components_active': active_components,
                        'covariance_type': covariance_type,
                        'silhouette': silhouette,
                        'log_likelihood': log_likelihood,
                        'labels': labels,
                        'weights': bgmm.weights_,
                        'bgmm_model': bgmm,
                        'converged': bgmm.converged_
                    })
                    
            except Exception as e:
                continue
    
    return bgmm_results

def analyze_probabilistic_uncertainty(gmm_model, X):
    """
    Analizza incertezza probabilistica
    Estratto da uncertainty analysis
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Probabilità predizione per ogni campione
    prediction_probs = gmm_model.predict_proba(X_scaled)
    
    # Analisi incertezza
    max_probs = np.max(prediction_probs, axis=1)
    mean_max_prob = np.mean(max_probs)
    
    # Entropia per misurare incertezza
    entropy = -np.sum(prediction_probs * np.log(prediction_probs + 1e-10), axis=1)
    mean_entropy = np.mean(entropy)
    
    # Campioni ambigui
    uncertain_samples = np.sum(max_probs < 0.7)  # Soglia arbitraria
    uncertainty_percentage = uncertain_samples / len(X_scaled) * 100
    
    return {
        'mean_max_probability': mean_max_prob,
        'mean_entropy': mean_entropy,
        'uncertain_samples': uncertain_samples,
        'uncertainty_percentage': uncertainty_percentage,
        'prediction_probabilities': prediction_probs,
        'max_probabilities': max_probs
    }
