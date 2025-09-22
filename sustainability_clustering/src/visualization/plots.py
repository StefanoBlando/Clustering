"""
Visualization functions extracted from notebook modules
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

def create_performance_landscape(results_dict, save_path=None):
    """
    Crea performance landscape plot
    Estratto da MODULO ORGANIZZAZIONE FIGURE
    """
    algorithms = list(results_dict.keys())
    silhouettes = [results_dict[alg]['silhouette'] for alg in algorithms]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Colori per famiglia algoritmica
    family_colors = {
        'Hierarchical': 'steelblue',
        'K-Means': 'orange', 
        'MCA': 'green',
        'GMM': 'red',
        'Spectral': 'purple',
        'Factor': 'brown',
        'SOM': 'pink',
        'Fuzzy': 'gray',
        'DBSCAN': 'olive'
    }
    
    colors = []
    for alg in algorithms:
        for family, color in family_colors.items():
            if family.lower() in alg.lower():
                colors.append(color)
                break
        else:
            colors.append('black')
    
    bars = ax.bar(range(len(algorithms)), silhouettes, color=colors, alpha=0.7, edgecolor='black')
    
    # Annotazioni performance
    for i, (bar, sil) in enumerate(zip(bars, silhouettes)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{sil:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Clustering Algorithm', fontweight='bold')
    ax.set_ylabel('Silhouette Score', fontweight='bold') 
    ax.set_title('Algorithm Performance Landscape\nSilhouette Scores Across Methodological Families', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(algorithms)))
    ax.set_xticklabels(algorithms, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Soglie qualità
    ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='Good threshold')
    ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='Acceptable threshold')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        return save_path
    else:
        plt.show()
        return fig

def create_dendrogram_plot(linkage_matrix, labels=None, save_path=None):
    """
    Crea dendrogramma
    Estratto da MODULO 5 dendrogramma
    """
    from scipy.cluster.hierarchy import dendrogram
    
    plt.figure(figsize=(20, 10))
    dendr = dendrogram(linkage_matrix,
                       labels=labels,
                       leaf_rotation=90,
                       leaf_font_size=6,
                       color_threshold=0.7*linkage_matrix[:, 2].max())
    
    plt.title('Hierarchical Clustering Dendrogram', fontsize=16, fontweight='bold')
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        plt.show()

def create_mca_biplot(coordinates, category_coords, explained_variance, save_path=None):
    """
    Crea MCA biplot
    Estratto da MODULO 6 MCA biplot
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Scatter plot osservazioni
    ax.scatter(coordinates[:, 0], coordinates[:, 1], alpha=0.6, s=30, color='lightblue', edgecolors='navy')
    
    # Plot categorie come frecce
    if category_coords is not None:
        for i, coord in enumerate(category_coords[:20]):  # Prime 20 per leggibilità
            ax.arrow(0, 0, coord[0], coord[1], head_width=0.05, head_length=0.05, 
                    fc='red', ec='red', alpha=0.7)
            ax.text(coord[0]*1.1, coord[1]*1.1, f'Cat_{i}', fontsize=8, ha='center')
    
    ax.set_xlabel(f'Dimension 1 ({explained_variance[0]*100:.1f}% variance)', fontweight='bold')
    ax.set_ylabel(f'Dimension 2 ({explained_variance[1]*100:.1f}% variance)', fontweight='bold')
    ax.set_title('MCA Biplot: Observations and Categories', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        plt.show()

def create_cluster_profiles_heatmap(profiles_df, cluster_labels, save_path=None):
    """
    Crea heatmap profili cluster
    Estratto da vari moduli profiling
    """
    plt.figure(figsize=(12, 8))
    
    sns.heatmap(profiles_df, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                xticklabels=profiles_df.columns,
                yticklabels=cluster_labels,
                cbar_kws={'label': 'Mean Value (Standardized)'})
    
    plt.title('Cluster Profiles Heatmap', fontsize=14, fontweight='bold')
    plt.ylabel('Clusters', fontweight='bold')
    plt.xlabel('Variables', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        plt.show()

def create_stability_analysis_plot(bootstrap_results, save_path=None):
    """
    Crea plot analisi stabilità
    Estratto da bootstrap validation modules
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Distribution ARI
    ax1.hist(bootstrap_results['bootstrap_aris'], bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(bootstrap_results['mean_ari'], color='red', linestyle='--', 
               label=f'Mean: {bootstrap_results["mean_ari"]:.3f}')
    ax1.axvline(bootstrap_results['ci_lower'], color='orange', linestyle='--', alpha=0.7)
    ax1.axvline(bootstrap_results['ci_upper'], color='orange', linestyle='--', alpha=0.7,
               label=f'95% CI: [{bootstrap_results["ci_lower"]:.3f}, {bootstrap_results["ci_upper"]:.3f}]')
    
    ax1.set_xlabel('Adjusted Rand Index')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Bootstrap Stability Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot per confronto
    ax2.boxplot(bootstrap_results['bootstrap_aris'], patch_artist=True,
               boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax2.set_ylabel('Adjusted Rand Index')
    ax2.set_title('Stability Summary')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Clustering Stability Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        plt.show()

def create_consensus_heatmap(consensus_matrix, save_path=None):
    """
    Crea consensus matrix heatmap
    Estratto da ensemble methods
    """
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(consensus_matrix, cmap='viridis', square=True,
                cbar_kws={'label': 'Consensus Score'})
    
    plt.title('Consensus Matrix Heatmap\nCross-Algorithm Agreement', 
             fontsize=14, fontweight='bold')
    plt.xlabel('Observations')
    plt.ylabel('Observations')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        plt.show()
