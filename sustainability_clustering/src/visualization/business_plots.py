"""
Business-focused visualization functions
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def create_executive_dashboard(clustering_results, save_path=None):
    """
    Crea executive dashboard
    Estratto da executive summary visualization
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel A: Performance ranking
    methods = list(clustering_results.keys())
    performances = [clustering_results[m]['silhouette'] for m in methods]
    
    bars = ax1.barh(range(len(methods)), performances, color='steelblue', alpha=0.8)
    ax1.set_yticks(range(len(methods)))
    ax1.set_yticklabels(methods)
    ax1.set_xlabel('Silhouette Score')
    ax1.set_title('A. Methodological Performance Ranking')
    ax1.grid(axis='x', alpha=0.3)
    
    # Annotazioni
    for bar, perf in zip(bars, performances):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{perf:.3f}', va='center', fontweight='bold')
    
    # Panel B: Variance explained breakdown
    variance_sources = ['Demographic', 'Geographic', 'Attitudinal', 'Other']
    variance_values = [73.4, 12.7, 1.9, 12.0]  # Da notebook analysis
    
    wedges, texts, autotexts = ax2.pie(variance_values, labels=variance_sources, autopct='%1.1f%%',
                                      colors=['lightcoral', 'lightblue', 'lightgreen', 'lightyellow'])
    ax2.set_title('B. Between-Cluster Variance Explained')
    
    # Panel C: Cluster size distribution (esempio per metodo migliore)
    cluster_sizes = [36.4, 27.8, 35.8]  # Da optimal solution
    cluster_labels = ['Professional\nMales', 'Young\nStudents', 'Mature\nWomen']
    
    bars3 = ax3.bar(cluster_labels, cluster_sizes, color=['lightblue', 'lightgreen', 'lightcoral'], alpha=0.8)
    ax3.set_ylabel('Percentage (%)')
    ax3.set_title('C. Optimal Segmentation (MCA K-Means)')
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, size in zip(bars3, cluster_sizes):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{size}%', ha='center', fontweight='bold')
    
    # Panel D: Universal barriers
    barriers = ['High Costs', 'Limited\nAvailability', 'Inconvenience', 'Information\nLack']
    barrier_pcts = [67.5, 28.1, 38.4, 44.7]  # Da analysis
    
    bars4 = ax4.bar(barriers, barrier_pcts, color='orange', alpha=0.8)
    ax4.set_ylabel('Percentage Reporting (%)')
    ax4.set_title('D. Universal Economic Barriers')
    ax4.grid(axis='y', alpha=0.3)
    
    for bar, pct in zip(bars4, barrier_pcts):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{pct}%', ha='center', fontweight='bold')
    
    plt.suptitle('Executive Summary Dashboard: Multi-Algorithm Sustainability Clustering', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        return save_path
    else:
        plt.show()

def create_strategic_roadmap(cluster_profiles, save_path=None):
    """
    Crea strategic roadmap visualization
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Tre segmenti con positioning
    segments = {
        'Professional Males\n(36.4%)': {
            'pos': (2, 7),
            'color': 'lightblue',
            'strategies': ['B2B Networks', 'ROI Focus', 'Premium Positioning']
        },
        'Young Students\n(27.8%)': {
            'pos': (6, 7),
            'color': 'lightgreen', 
            'strategies': ['Social Media', 'Innovation', 'Affordability']
        },
        'Mature Women\n(35.8%)': {
            'pos': (10, 7),
            'color': 'lightcoral',
            'strategies': ['Community', 'Practicality', 'Value Focus']
        }
    }
    
    # Core unified messaging al centro
    ax.add_patch(plt.Rectangle((5.5, 3.5), 3, 2, facecolor='gold', alpha=0.8, edgecolor='black'))
    ax.text(7, 4.5, 'UNIFIED SUSTAINABILITY\nMESSAGING CORE', ha='center', va='center',
           fontsize=12, fontweight='bold')
    
    # Segmenti
    for segment, info in segments.items():
        x, y = info['pos']
        
        # Box segmento
        ax.add_patch(plt.Rectangle((x-1, y-1), 2, 1.5, facecolor=info['color'], 
                                  alpha=0.7, edgecolor='black'))
        ax.text(x, y-0.2, segment, ha='center', va='center', fontweight='bold')
        
        # Strategie
        for i, strategy in enumerate(info['strategies']):
            ax.text(x, y-1.5-i*0.3, f'• {strategy}', ha='center', fontsize=9)
        
        # Freccia verso centro
        ax.annotate('', xy=(7, 5.5), xytext=(x, y-1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))
    
    # Universal barriers box
    ax.add_patch(plt.Rectangle((5.5, 0.5), 3, 1.5, facecolor='orange', alpha=0.6, edgecolor='black'))
    ax.text(7, 1.25, 'UNIVERSAL ECONOMIC\nBARRIER MITIGATION', ha='center', va='center',
           fontsize=11, fontweight='bold')
    
    # Freccia dal core ai barriers
    ax.annotate('', xy=(7, 2), xytext=(7, 3.5),
               arrowprops=dict(arrowstyle='->', lw=3, color='red'))
    
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Strategic Roadmap: Demographic-Adapted Sustainability Marketing', 
                fontsize=16, fontweight='bold', pad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        return save_path
    else:
        plt.show()

def create_sustainability_paradox_plot(save_path=None):
    """
    Visualizza il paradosso sostenibilità
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: Demographic dominance
    demographics = ['Age', 'Occupation', 'Education', 'Gender', 'Income']
    variance_explained = [23.4, 46.7, 41.2, 38.9, 28.1]
    
    bars1 = ax1.bar(demographics, variance_explained, 
                   color='steelblue', alpha=0.8, edgecolor='navy')
    ax1.set_ylabel('Variance Explained (%)', fontweight='bold')
    ax1.set_title('A. Robust Demographic Segmentation\n(Total: 73.4% variance)', fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars1, variance_explained):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val}%', ha='center', fontweight='bold')
    
    # Panel B: Attitudinal homogeneity
    attitudes = ['Env.\nAwareness', 'Perceived\nEfficacy', 'Daily\nBehaviors', 
                'Home\nMgmt', 'Future\nIntentions']
    f_statistics = [0.089, 0.245, 0.123, 0.198, 0.112]
    
    bars2 = ax2.bar(attitudes, f_statistics, 
                   color='lightcoral', alpha=0.8, edgecolor='darkred')
    ax2.set_ylabel('F-statistic', fontweight='bold')
    ax2.set_title('B. Complete Attitudinal Homogeneity\n(All p > 0.05)', fontweight='bold')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Significance threshold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, f_val in zip(bars2, f_statistics):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'F={f_val:.3f}', ha='center', fontsize=9)
    
    plt.suptitle('The Sustainability Paradox: Demographic vs Attitudinal Patterns', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        return save_path
    else:
        plt.show()
