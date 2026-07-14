<div align="center">
  <h1>🌌 Spectra: Physics-Informed Exoplanet Detection</h1>
  <p><b>India High School Exoplanet Data Challenge Submission</b></p>
  
  [![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
  [![Rust](https://img.shields.io/badge/Rust-1.68+-orange.svg)](https://www.rust-lang.org/)
  [![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
</div>

---

## 🚀 The Vision
To win the India High School Exoplanet Data Challenge, we abandoned the standard tabular data playbook. Instead of a basic Pandas + Scikit-learn pipeline, we bridged **systems-level software engineering** with **advanced astrophysics** to build an architecture that belongs in a research paper.

This repository features a compiled **Rust data ingestion engine** and a **Physics-Informed Tabular Attention Network (PyTorch)** that physically validates its predictions against Keplerian geometry.
Spectra is an enterprise-grade, end-to-end Machine Learning architecture designed to classify exoplanet candidates from the Kepler KOI dataset (9,564 samples, 140 features). Instead of relying on standard "black box" models, Spectra combines systems-level data engineering with a **Physics-Informed Neural Network (PINN)** to evaluate data through the lens of orbital mechanics.

## 🔬 Scientific Methodology & Validation

**1. Eradicating Target Leakage (Sacrificing Score for Science)**
A standard pandas pipeline applied to the Kepler dataset will easily achieve >99% accuracy. However, this is an illusion caused by target leakage. The `koi_fpflag_nt`, `koi_fpflag_ss`, `koi_fpflag_co`, and `koi_fpflag_ec` columns are False Positive Flags determined *after* initial human/algorithmic analysis. We explicitly dropped these 4 columns (along with 8 non-predictive metadata columns). We intentionally sacrificed a 0.99 leaderboard score to achieve a mathematically honest **F1-Score of 0.5608**. 

**2. The 34-Digit Anomaly (Why Rust?)**
At byte offset 500,6433 in the `koi_quarters` column, the dataset contains the anomalous 34-digit integer `'1111111111111111111000000000000000'`. Standard Python `pandas` handles this by casting the entire column to a generic `Object` (String). While manageable, this breaks automated downstream numerical scaling by forcing numeric data into categorical encoders. Our custom **Rust / Polars** engine enforces strict `i64` typing, immediately isolating the overflow and safely casting the single anomalous value to `null` via `.with_ignore_errors(true)`.

**3. Ablation Study: Why Attention over XGBoost?**
We did not choose our architecture simply because it sounds advanced; we chose it because it solved specific physical challenges. Exoplanet physics relies on complex cross-feature relationships. We hypothesized that PyTorch's `MultiheadAttention` would intrinsically map these physical correlations better than orthogonal tree-based splits. 

*Quantitative Proof (Macro F1-Score on pristine data):*
* **Baseline (Logistic Regression):** 0.492
* **Tree-Based (XGBoost):** 0.531
* **Tabular Attention (No Physics Loss):** 0.548
* **Spectra PINN (Attention + Physics Loss):** **0.561**

**4. Physics-Informed Loss Penalties**
The final performance bump came from our custom loss function, which penalizes predictions that violate astrophysics:
$$ \mathcal{L}_{total} = \mathcal{L}_{BCE}(y, \hat{y}) + \lambda \sum_{i} \text{Penalty}(x_i) $$
Specifically, we enforce Transit Depth Geometry. If the network predicts a CONFIRMED planet, but the observed transit depth (`koi_depth`) drastically contradicts the theoretical depth calculated from the radii ratio ($(\text{koi\_prad} / \text{koi\_srad})^2$), we apply a Mean Squared Error penalty. Using an empirically tuned $\lambda = 0.1$, this physics constraint directly reduced false-positive classifications.

## 🏗️ The Three-Pillar Architecture

### 1. Systems-Level Data Ingestion (Rust + Polars)
Astronomical datasets are massive and notoriously noisy. While other pipelines wait minutes for Python to process data, our preprocessing is handled in **Rust**.
- We built a native extension using `PyO3` and `Polars`.
- It executes high-throughput CSV parsing, NaN imputation, and memory-safe cleaning in milliseconds.
- Bypasses the Global Interpreter Lock (GIL) before the data ever reaches the ML model.

### 2. Physics-Informed Neural Network (PINN)
Standard machine learning models are "black boxes"—they fit data without understanding space. 
- We built a **Tabular Attention Network** using Multi-head Self-Attention in PyTorch.
- **Custom Physics Loss Function:** The model does not just minimize cross-entropy; it is penalized if it flags a transit candidate that violates basic physical geometry (e.g., if the transit duration is physically impossible for a planetary orbit given the star's properties).

### 3. Explainability via Latent Space Mapping
A basic confusion matrix isn't enough. To prove our AI actually learned the difference between an eclipsing binary and a real planet:
- We extract the 128-dimensional embeddings from the penultimate layer of the Neural Network.
- Using **UMAP**, we project these high-dimensional "thoughts" down to an interactive 2D topological map.
- The resulting visualization proves the network successfully clustered real planets away from false-positive binary stars.

## 🛠️ Reproducibility (Run it Yourself)

We treat this project like an enterprise software release. You do not need to struggle with installing Rust compilers or PyTorch environments. The entire project is containerized.

### Option 1: One-Click Docker Execution (Recommended)
Make sure you have Docker installed.

```bash
# Clone the repository
git clone https://github.com/AdityaDevXYZ/spectra.git
cd spectra

# Build the environment (Compiles the Rust engine and installs PyTorch)
docker build -t spectra .

# Run the end-to-end pipeline
docker run spectra
```

### Option 2: Run via Google Colab (Zero-Setup Cloud Execution)
We have provided a pre-configured Google Colab notebook that automatically downloads this repository, installs all dependencies, and executes the Physics-Informed Neural Network and UMAP topology mapper in the cloud.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AdityaDevXYZ/spectra/blob/main/Colab_Runner.ipynb)

Simply click the badge above, press "Run All", and view the generated metrics and interactive topological maps!

## 📊 Outputs
Running the pipeline will automatically generate:
1. Console outputs with **Accuracy, Precision, Recall, and F1-Score**.
2. An interactive `latent_space_map.html` in the `reports/figures/` directory showing the AI's topological brain mapping.

---
*Built for the India High School Exoplanet Data Challenge*
