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
