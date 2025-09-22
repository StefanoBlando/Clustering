"""
Self-Organizing Maps functions extracted from notebook
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class SelfOrganizingMap:
    """
    SOM implementation estratto dai moduli neural
    """
    
    def __init__(self, grid_size=(6, 6), learning_rate=0.5, neighborhood_radius=3.0):
        self.grid_size = grid_size
        self.learning_rate = learning_rate
        self.neighborhood_radius = neighborhood_radius
        self.weights = None
        
    def initialize_weights(self, n_features):
        """Inizializza pesi random"""
        self.n_features = n_features
        self.weights = np.random.random((self.grid_size[0], self.grid_size[1], n_features))
        
    def find_bmu(self, x):
        """Find Best Matching Unit"""
        distances = np.sum((self.weights - x) ** 2, axis=2)
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_idx
    
    def update_weights(self, x, bmu_idx, epoch, total_epochs):
        """Update weights con decadimento"""
        # Decadimento learning rate
        lr = self.learning_rate * (1 - epoch / total_epochs)
        
        # Decadimento neighborhood radius  
        radius = self.neighborhood_radius * (1 - epoch / total_epochs)
        
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                # Distanza dal BMU
                dist = np.sqrt((i - bmu_idx[0])**2 + (j - bmu_idx[1])**2)
                
                if dist <= radius:
                    # Neighborhood function (Gaussian)
                    influence = np.exp(-(dist**2) / (2 * radius**2))
                    
                    # Update weights
                    self.weights[i, j] += lr * influence * (x - self.weights[i, j])
    
    def fit(self, X, epochs=1000):
        """Training SOM - estratto da moduli neural"""
        self.initialize_weights(X.shape[1])
        
        for epoch in range(epochs):
            for x in X:
                bmu_idx = self.find_bmu(x)
                self.update_weights(x, bmu_idx, epoch, epochs)
                
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}")
        
        return self
    
    def predict(self, X):
        """Predict cluster assignments"""
        labels = []
        for x in X:
            bmu_idx = self.find_bmu(x)
            # Converti coordinate 2D in label 1D
            label = bmu_idx[0] * self.grid_size[1] + bmu_idx[1]
            labels.append(label)
        return np.array(labels)

def perform_som_clustering(X, grid_size=(6, 6), epochs=1000):
    """
    SOM clustering completo
    Estratto da moduli neural
    """
    # Standardizzazione
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Training SOM
    som = SelfOrganizingMap(grid_size=grid_size)
    som.fit(X_scaled, epochs=epochs)
    
    # Predizioni
    som_labels = som.predict(X_scaled)
    
    # Meta-clustering sui neuroni attivati
    unique_labels = np.unique(som_labels)
    if len(unique_labels) > 1:
        # K-means sui centroidi neuroni
        neuron_centers = []
        for label in unique_labels:
            i, j = divmod(label, grid_size[1])
            neuron_centers.append(som.weights[i, j])
        
        if len(neuron_centers) >= 3:
            neuron_centers = np.array(neuron_centers)
            kmeans_meta = KMeans(n_clusters=3, random_state=42)
            meta_labels = kmeans_meta.fit_predict(neuron_centers)
            
            # Mappa label originali a meta-clusters
            meta_mapping = dict(zip(unique_labels, meta_labels))
            final_labels = np.array([meta_mapping[label] for label in som_labels])
            
            # Calcola silhouette
            if len(np.unique(final_labels)) > 1:
                silhouette = silhouette_score(X_scaled, final_labels)
            else:
                silhouette = 0.0
                
            return {
                'som_model': som,
                'som_labels': som_labels,
                'meta_labels': final_labels,
                'silhouette': silhouette,
                'n_neurons_active': len(unique_labels)
            }
    
    return {
        'som_model': som,
        'som_labels': som_labels,
        'meta_labels': som_labels,
        'silhouette': 0.0,
        'n_neurons_active': len(unique_labels)
    }
