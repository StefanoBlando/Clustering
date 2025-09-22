# Multi-Algorithm Sustainability Clustering Analysis

## Overview

Comprehensive multi-algorithm clustering analysis of environmental sustainability behaviors among 151 Italian consumers. This project systematically implements 15 different clustering methodologies across 8 algorithmic families to identify robust patterns in consumer environmental segmentation.

**Academic Context:** Master's program project from the Clustering Analysis Laboratory module at University of Rome Tor Vergata, Department of Economics and Finance.

## Key Findings

**Best Algorithm:** Hierarchical clustering with average linkage (Silhouette = 0.606)

**Main Discovery:** The "Sustainability Paradox" - robust demographic segmentation coexists with complete attitudinal homogeneity

**Three Validated Segments:**
- **Professional Educated Males** (36.4%): Resource-rich efficiency seekers
- **Young Students** (27.8%): Future-oriented innovation adopters  
- **Mature Educated Women** (35.8%): Practical family-oriented adopters

**Universal Finding:** Economic barriers dominate across all segments (67.5% report cost concerns)

## Repository Structure

```
sustainability-clustering-analysis/
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
├── .gitignore
├── data/
│   ├── raw/
│   │   └── Questionario_Sostenibilita_1_1.xlsx
│   └── processed/
│       └── (generated during analysis)
├── notebooks/
│   ├── 01_preprocessing_base.ipynb
│   ├── 02_hierarchical_clustering.ipynb
│   ├── 03_mca_analysis.ipynb
│   ├── 04_lca_analysis.ipynb
│   ├── 05_factor_analysis.ipynb
│   ├── 06_ensemble_methods.ipynb
│   ├── 07_business_interpretation.ipynb
│   └── 08_advanced_algorithms.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── hierarchical.py
│   ├── mca.py
│   ├── factor_analysis.py
│   ├── som.py
│   ├── fuzzy.py
│   ├── gaussian_mixture.py
│   ├── advanced_algorithms.py
│   ├── ensemble_methods.py
│   ├── business_interpretation.py
│   ├── plots.py
│   ├── business_plots.py
│   └── validation/
│       ├── __init__.py
│       ├── metrics.py
│       └── statistical_tests.py
├── reporting/
│   ├── executive_summary.md
│   ├── methodology_report.md
│   ├── business_insights.md
│   └── technical_appendix.md
├── scripts/
│   ├── run_preprocessing.py
│   ├── run_clustering.py
│   └── generate_report.py
└── tests/
    ├── __init__.py
    ├── test_preprocessing.py
    └── test_clustering.py
```

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/[username]/sustainability-clustering-analysis.git
cd sustainability-clustering-analysis

# Create conda environment
conda env create -f environment.yml
conda activate sustainability-clustering

# Or use pip
pip install -r requirements.txt
```

### Data Setup

1. Place `Questionario_Sostenibilita_1_1.xlsx` in `data/raw/`
2. Run preprocessing: `python scripts/run_preprocessing.py`
3. Execute analysis: `python scripts/run_clustering.py`

### Core Analysis

```python
# Load and preprocess data
from src.data_loader import load_sustainability_dataset, preprocess_demographics
from src.hierarchical import perform_hierarchical_clustering

# Load dataset
df = load_sustainability_dataset('data/raw/Questionario_Sostenibilita_1_1.xlsx')
df_processed = preprocess_demographics(df)

# Run best performing algorithm
results = perform_hierarchical_clustering(X_scaled, method='average', k_range=range(2, 6))
```

## Methodology

### 15 Algorithms Across 8 Families
1. **Partitional:** K-Means, K-Means on MCA coordinates
2. **Hierarchical:** Ward, Average, Complete linkage
3. **Density-based:** DBSCAN with parameter optimization
4. **Spectral:** RBF kernel, k-NN affinity, polynomial
5. **Probabilistic:** Gaussian Mixture Models, Bayesian GMM
6. **Neural:** Self-Organizing Maps
7. **Fuzzy:** Fuzzy C-Means with membership analysis
8. **Ensemble:** Consensus clustering, weighted voting

### Validation Framework
- Bootstrap stability analysis (100 replications)
- Permutation testing (1000 permutations)
- Cross-methodological concordance
- Survey-specific validation (measurement invariance, IRT analysis)

## Main Results

### Algorithm Performance Ranking

| Rank | Algorithm | Silhouette | Stability (ARI) | Interpretation |
|------|-----------|------------|-----------------|----------------|
| 1 | Hierarchical Average | 0.606 | 0.847 | Exceptional |
| 2 | Ensemble Consensus | 0.491 | 0.712 | High |
| 3 | MCA K-Means | 0.405 | 0.673 | High |
| 4 | GMM Tied | 0.422 | 0.634 | Medium-High |
| 5 | Spectral RBF | 0.421 | 0.598 | Medium |

### The Sustainability Paradox

**Demographic Dominance:** 73.4% of between-cluster variance explained by demographics  
**Attitudinal Homogeneity:** F = 0.83, p = 0.694 (no significant differences)  
**Universal Economic Barriers:** 67.5% report cost concerns across all segments

## Business Applications

### Strategic Recommendations
- **Demographic-adapted targeting** with unified sustainability messaging
- **Universal economic barrier mitigation** through structural interventions
- **Progressive engagement frameworks** acknowledging behavioral ambiguity

### Policy Implications
- Economic incentives more effective than attitude change campaigns
- Demographic-specific implementation of sustainability programs
- Focus on structural constraints rather than individual preferences

## Usage Examples

### Run Complete Analysis
```python
# Execute full pipeline
python scripts/run_clustering.py --method all --output results/

# Generate business insights
python scripts/generate_report.py --format executive
```

### Custom Analysis
```python
from src.ensemble_methods import consensus_clustering
from src.advanced_algorithms import spectral_clustering_comprehensive

# Ensemble analysis
algorithms = {'kmeans': KMeans(n_clusters=3), 'hierarchical': AgglomerativeClustering(n_clusters=3)}
consensus_result = consensus_clustering(X, algorithms, n_runs=20)

# Advanced algorithms
spectral_results = spectral_clustering_comprehensive(X, affinity_types=['rbf', 'nearest_neighbors'])
```


## Dataset Information

- **Sample:** 151 Italian consumers
- **Variables:** 29 survey items (demographics + environmental attitudes/behaviors)
- **Geographic:** Concentrated in Central Italy (65.6%)
- **Limitations:** Educational bias (57.6% university vs 24% national average)

## Dependencies

- Python 3.9+
- scikit-learn 1.3.0
- pandas 2.0.3
- numpy 1.24.3
- matplotlib 3.7.2
- seaborn 0.12.2
- factor-analyzer 0.4.1
- prince 0.7.1 (MCA)

## Citation

```bibtex
@article{blando2024sustainability,
  title={Multi-Algorithm Sustainability Clustering Analysis: A Comprehensive Empirical Investigation of Consumer Environmental Behavior Segmentation},
  author={Blando, Stefano},
  year={2024},
  institution={University of Rome Tor Vergata},
  course={Clustering Analysis Laboratory},
  supervisor={Prof. Furio Camillo}
}
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/algorithm-name`)
3. Commit changes (`git commit -am 'Add new clustering algorithm'`)
4. Push to branch (`git push origin feature/algorithm-name`)
5. Create Pull Request

## Contact

**Author:** Stefano Blando  
**Institution:** University of Rome Tor Vergata  
**Course:** Clustering Analysis Laboratory  
**Supervisor:** Prof. Furio Camillo

## Acknowledgments

- Prof. Furio Camillo (Course Supervisor)
- University of Rome Tor Vergata, Department of Economics and Finance
- Clustering Analysis Laboratory participants
- Survey participants (151 Italian consumers)
- Open-source Python ecosystem (scikit-learn, pandas, matplotlib)

---

**Methodological Innovation:** First comprehensive multi-algorithm validation in sustainability behavior research  
**Academic Contribution:** Advanced laboratory project in clustering methodologies  
**Practical Impact:** Evidence-based framework for demographic-adapted sustainability marketing
