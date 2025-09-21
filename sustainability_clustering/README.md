# Multi-Algorithm Sustainability Clustering Analysis

## Overview

This repository contains a comprehensive multi-algorithm clustering analysis of environmental sustainability behaviors among 151 Italian consumers. The study systematically implements 15 different clustering methodologies across 8 algorithmic families to identify robust patterns in consumer environmental segmentation.

## Key Findings

**🏆 Best Algorithm:** Hierarchical clustering with average linkage (Silhouette = 0.606)

**📊 Main Discovery:** The "Sustainability Paradox" - robust demographic segmentation coexists with complete attitudinal homogeneity

**👥 Three Validated Segments:**
- **Professional Educated Males** (36.4%): Resource-rich efficiency seekers
- **Young Students** (27.8%): Future-oriented innovation adopters  
- **Mature Educated Women** (35.8%): Practical family-oriented adopters

**💰 Universal Finding:** Economic barriers dominate across all segments (67.5% report cost concerns)

## Dataset

- **Sample:** 151 Italian consumers
- **Variables:** 29 survey items (demographics + environmental attitudes/behaviors)
- **Geographic:** Concentrated in Central Italy
- **Bias:** Over-representation of educated, urban, younger populations

## Methodology

### 8 Algorithmic Families Tested
1. **Partitional:** K-Means, K-Means on MCA coordinates
2. **Hierarchical:** Ward, Average, Complete linkage
3. **Density-based:** DBSCAN
4. **Spectral:** RBF kernel, k-NN affinity
5. **Probabilistic:** Gaussian Mixture Models, Bayesian GMM
6. **Neural:** Self-Organizing Maps
7. **Fuzzy:** Fuzzy C-Means
8. **Ensemble:** Consensus clustering

### Validation Framework
- Bootstrap stability analysis (100 replications)
- Permutation testing (1000 permutations)
- Cross-methodological concordance
- Survey-specific validation (measurement invariance, social desirability bias)

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
# Run complete analysis pipeline
from src.clustering.ensemble import run_complete_analysis

results = run_complete_analysis(
    data_path='data/raw/Questionario_Sostenibilita_1_1.xlsx',
    output_dir='results/'
)

# Best performing method
print(f"Best algorithm: {results['best_algorithm']}")
print(f"Silhouette score: {results['best_silhouette']:.3f}")
```

## Repository Structure

```
├── notebooks/           # Jupyter notebooks for each analysis module
├── src/                # Core analysis code
├── data/               # Raw, processed, and result datasets
├── figures/            # Generated visualizations
├── reports/            # Analysis reports and findings
└── scripts/            # Execution scripts
```

## Key Notebooks

1. **01_preprocessing_base.ipynb** - Data cleaning and feature engineering
2. **03_hierarchical_clustering.ipynb** - Best performing method
3. **04_mca_analysis.ipynb** - Multiple Correspondence Analysis
4. **06_ensemble_methods.ipynb** - Multi-algorithm consensus
5. **99_complete_analysis.ipynb** - Full pipeline and business insights

## Main Results

### Algorithm Performance Ranking

| Rank | Algorithm | Silhouette | Stability (ARI) | Interpretation |
|------|-----------|------------|-----------------|----------------|
| 1 | Hierarchical Average | 0.606 | 0.847 | Exceptional |
| 2 | Ensemble Consensus | 0.491 | 0.712 | High |
| 3 | MCA K-Means | 0.405 | 0.673 | High |
| 4 | GMM Tied | 0.422 | 0.634 | Medium-High |
| 5 | Spectral RBF | 0.421 | 0.598 | Medium |

### Demographic vs Attitudinal Variance

- **Demographic variables:** 73.4% of between-cluster variance
- **Attitudinal variables:** 1.9% of between-cluster variance
- **Statistical test:** F = 0.83, p = 0.694 (no significant attitudinal differences)

### Business Implications

**Strategic Recommendation:** Shift from values-based to demographic-adapted segmentation with unified sustainability messaging

**Policy Insight:** Universal economic barriers suggest structural interventions (subsidies, tax incentives) more effective than attitude change campaigns

## Research Applications

This methodology is transferable to:
- Health behavior research
- Technology adoption studies  
- Financial decision-making analysis
- Social behavior segmentation
- Any domain with attitude-behavior gaps

## Citation

```bibtex
@thesis{blando2024sustainability,
  title={Multi-Algorithm Sustainability Clustering Analysis: A Comprehensive Empirical Investigation of Consumer Environmental Behavior Segmentation},
  author={Blando, Stefano},
  year={2024},
  institution={University of Rome Tor Vergata},
  type={Master's Thesis}
}
```

## Dependencies

- Python 3.9+
- scikit-learn 1.3.0
- pandas 2.0.3
- numpy 1.24.3
- matplotlib 3.7.2
- seaborn 0.12.2
- scipy 1.11.1

## License

GNU Affero General Public License v3.0 License - see LICENSE file for details

## Contact

**Author:** Stefano Blando  
**Institution:** University of Rome Tor Vergata  
**Email:** [email]  
**LinkedIn:** [profile]

## Acknowledgments

- Prof. Furio Camillo (Supervisor)
- University of Rome Tor Vergata, Department of Economics and Finance
- Survey participants (151 Italian consumers)

---

**🔬 Methodological Innovation:** First comprehensive multi-algorithm validation in sustainability behavior research

**📈 Practical Impact:** Evidence-based framework for demographic-adapted sustainability marketing

**🌱 Environmental Relevance:** Insights for accelerating sustainable behavior adoption at scale
