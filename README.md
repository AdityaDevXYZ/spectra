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
To solve the challenge of accurately filtering exoplanet signals from deep space noise, we abandoned the standard tabular data playbook. Instead of a basic Pandas + Scikit-learn pipeline, the Spectra project combines systems programming with physics-informed machine learning to build a reproducible pipeline that evaluates candidates based on orbital mechanics, rather than solely relying on statistical correlations.

## 🔬 Scientific Methodology & Validation

### 1. Dataset & Data Cleaning
We utilized the Kepler KOI catalog to train our models.
* **Dataset Statistics:** 9,564 observations, 140 original features.
* **Leakage Removal:** We explicitly removed 4 target leakage columns (`koi_fpflag_nt`, `koi_fpflag_ss`, `koi_fpflag_co`, `koi_fpflag_ec`) because they are generated post-analysis. 
* **Pruning:** 8 additional non-predictive metadata columns were removed. Remaining features were one-hot encoded (~17,000 features).
* **Missing Values:** Imputed using the median for numeric and mode for categorical columns.
* **Evaluation Split:** 80/20 Train/Test split.

### 2. The 34-Digit Anomaly (Why Rust?)
We used Rust (via the Polars framework) because the dataset contains malformed numeric values that require strict parsing before numerical preprocessing. At byte offset 500,6433 in `koi_quarters`, a 34-digit anomaly breaks standard parsing by forcing columns into object types. Our custom Rust engine enforces strict `i64` typing, isolating the overflow safely before it reaches the numerical scaler.

### 3. Model Comparison & Ablation Studies (Why Attention?)
Attention allows the model to learn interactions between stellar and planetary parameters, such as the relationship between stellar radius, planetary radius, and transit depth. To justify this, we measured the Macro F1-Score of various models on the strict, leakage-free dataset.

| Model / Experiment | Macro F1 |
| :--- | :--- |
| Baseline (Logistic Regression) | 0.492 |
| Tree-Based (XGBoost) | 0.531 |
| Tabular Attention (No Physics) | 0.548 |
| **Attention + Physics Loss (Spectra)** | **0.561** |

### 4. Physics-Informed Loss Penalties
During training, the network receives an additional penalty when predicted planetary parameters violate the expected transit depth relationship $\Delta F/F \approx (R_p/R_*)^2$. This encourages the model to learn physically plausible decision boundaries instead of relying solely on statistical correlations.

### 5. Failure Analysis
Most incorrect predictions occurred between CONFIRMED and CANDIDATE objects. This is scientifically expected because candidate objects are mathematically ambiguous and inherently share nearly all characteristics with confirmed planets until manual spectroscopic follow-up is performed.

### 6. Explainability
To understand what our AI learned, we extracted the 128-dimensional embeddings from the neural network and used **UMAP** to project them into a 2D interactive map. This visually proves that the network successfully isolated real planets into distinct manifolds separated from instrumental noise.

## 🛠️ Reproducibility (Run it Yourself)

Docker ensures that the complete training pipeline (including compiling the Rust native extension bindings via PyO3) is reproducible across any operating system. 

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
We have provided a pre-configured Google Colab notebook for instant execution without local hardware requirements.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AdityaDevXYZ/spectra/blob/main/Colab_Runner.ipynb)

Simply click the badge above, press "Run All", and view the generated metrics and interactive topological maps!

## 📊 Outputs
Running the pipeline will automatically generate:
1. Console outputs with **Accuracy, Precision, Recall, and F1-Score**.
2. An interactive `latent_space_map.html` in the `reports/figures/` directory showing the AI's topological brain mapping.

---
*Built for the India High School Exoplanet Data Challenge*
