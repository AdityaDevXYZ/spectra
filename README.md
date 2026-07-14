# Spectra: Physics-Informed Exoplanet Classification

## 1. Introduction
**Challenge:** Accurately filtering exoplanet signals from deep space noise in the Kepler dataset.
**Goal:** The project combines systems programming with physics-informed machine learning to build a reproducible pipeline that evaluates candidates based on orbital mechanics, rather than solely relying on statistical correlations.

## 2. Dataset & Data Cleaning
We utilized the Kepler KOI catalog to train our models.
* **Dataset Statistics:** 9,564 observations, 140 original features.
* **Leakage Removal:** We explicitly removed 4 target leakage columns (`koi_fpflag_nt`, `koi_fpflag_ss`, `koi_fpflag_co`, `koi_fpflag_ec`) because they are generated post-analysis. 
* **Pruning:** 8 additional non-predictive metadata columns removed.
* **Missing Values:** Imputed using the median for numeric and mode for categorical columns.
* **Evaluation Split:** 80/20 Train/Test split.

**Rust Preprocessing for Anomalies**
We used Rust (via the Polars framework) because the dataset contains malformed numeric values that require strict parsing. At byte offset 500,6433 in `koi_quarters`, a 34-digit anomaly breaks standard parsing by forcing columns into object types. Our Rust engine enforces strict `i64` typing, isolating the overflow safely before it reaches the numerical scaler.

## 3. Model Development
**Why Attention?**
Attention allows the model to learn interactions between stellar and planetary parameters, such as the relationship between stellar radius, planetary radius, and transit depth. We implemented a `TabularAttentionNetwork` using PyTorch.

**Physics-Informed Loss**
During training, the network receives an additional penalty when predicted planetary parameters violate the expected transit depth relationship $\Delta F/F \approx (R_p/R_*)^2$. This encourages the model to learn physically plausible decision boundaries instead of relying solely on statistical correlations.

## 4. Experimental Results
**Model Comparison & Ablation Studies**
To justify our architecture, we measured the Macro F1-Score of various models on the strict, leakage-free dataset.

| Model / Experiment | Macro F1 |
| :--- | :--- |
| Baseline (Logistic Regression) | 0.492 |
| Tree-Based (XGBoost) | 0.531 |
| Tabular Attention (No Physics) | 0.548 |
| **Attention + Physics Loss (Spectra)** | **0.561** |

**Ablation Study Summary:**
* **Physics Loss:** Increased F1 by ~0.013 by reducing physically impossible false positives.
* **UMAP Visualization:** No effect on accuracy, but drastically improves interpretability of the latent space.

**Failure Analysis**
Most incorrect predictions occurred between CONFIRMED and CANDIDATE objects. This is scientifically expected because candidate objects are mathematically ambiguous and inherently share nearly all characteristics with confirmed planets until manual spectroscopic follow-up is performed.

## 5. Explainability
To understand what our AI learned, we extracted the 128-dimensional embeddings from the neural network and used **UMAP** to project them into a 2D interactive map. This visually proves that the network successfully isolated real planets into distinct manifolds.

## 6. Reproducibility
We used Docker to ensure that the complete training pipeline (including compiling the Rust bindings) is reproducible across any operating system. We also provided a Google Colab notebook for instant execution.
