# Seismic-Based Oil Reservoir Characterization using Ensemble Learning & Deep Neural Networks

## 👥 Authors

| Name | Department |
|---|---|
| Devisha Bhargava | Computer Science & Engineering |
| Shivam Shaurya | Computer Science & Engineering |
| Kaustav Banerjee | Electrical & Electronics Engineering |

---

## 📄 Abstract

This project develops an **interpretable and uncertainty-aware machine learning framework** for predicting reservoir porosity from seismic-derived attributes. A hybrid pipeline combines a **Random Forest regressor** (for feature importance and explainability) with a **1D Convolutional Neural Network** (for capturing nonlinear seismic relationships), further enhanced by **Monte Carlo Dropout** for probabilistic uncertainty quantification. The framework achieves strong R² values and low RMSE, demonstrating reliability for seismic-based reservoir characterization.

---

## 🗂️ Repository Structure

```
reservoir-ml/
│
├── data/
│   └── synthetic_reservoir.csv      # Synthetic dataset with seismic attributes & porosity labels
│
├── models/
│   ├── random_forest.py             # Random Forest regressor with GridSearchCV tuning
│   ├── cnn_1d.py                    # 1D CNN model definition (PyTorch / Keras)
│   └── mc_dropout.py                # Monte Carlo Dropout inference wrapper
│
├── preprocessing/
│   └── feature_engineering.py       # Z-score normalization, noise injection, attribute generation
│
├── evaluation/
│   ├── metrics.py                   # R², RMSE computation utilities
│   └── uncertainty.py               # MC Dropout mean & std deviation aggregation
│
├── visualization/
│   ├── feature_importance.py        # Seismic attribute importance bar chart (Fig 3)
│   ├── correlation_heatmap.py       # Feature correlation heatmap (Fig 2)
│   ├── attribute_vs_porosity.py     # Scatter plots: RMS, Frequency, Amplitude vs Porosity (Figs 4–6)
│   └── mc_prediction_plot.py        # Prediction with uncertainty bands (Fig 1)
│
├── notebooks/
│   └── full_analysis.ipynb          # End-to-end walkthrough notebook
│
├── requirements.txt
└── README.md
```

---

## 🔬 Methodology

The pipeline follows four sequential stages:

```
Data Preparation          Random Forest              1D CNN                Uncertainty
& Feature Engineering  →  (Explainable ML Layer)  →  (Deep Learning Layer)  →  Quantification
                                                                              (MC Dropout)
     Z-score norm            Feature importance         Nonlinear attribute      Mean prediction
     Gaussian noise          High interpretability      interaction modeling     Std dev (confidence)
     Seismic attributes      Robust to noise            High representational    Confidence-aware
     generation              GridSearchCV tuning        capacity                 reservoir decisions
```

### Input Seismic Attributes

| Attribute | Role |
|---|---|
| **Amplitude** | Primary predictor — direct seismic response |
| **RMS** | Root Mean Square energy — related to porosity zones |
| **Frequency** | Spectral content — sensitive to fluid saturation |
| **Impedance** | Strongest inverse correlation with porosity |
| **Coherence** | Structural continuity — fault and fracture indicator |

### Target Variable
- **Porosity** — predicted across rock types: Sandstone, Shale, Limestone, and Fault zones

---

## 📊 Key Results

| Model | R² | RMSE |
|---|---|---|
| Random Forest (baseline) | High | Low |
| 1D CNN | Higher | Lower |
| 1D CNN + MC Dropout | High + Uncertainty bounds | Low |

- **Impedance** and **Amplitude** emerged as dominant predictors from feature importance analysis, consistent with established geophysical theory.
- MC Dropout uncertainty estimates aligned with physically complex regions (fault zones, heterogeneous lithologies).

### Sample Predictions

| Rock Type | Porosity | Permeability | Seismic Score | Amplitude | RMS | Predicted Porosity |
|---|---|---|---|---|---|---|
| Sandstone | 10.41 | 566.73 | 0.689 | 0.531 | 0.638 | 10.59 |
| Shale | 11.73 | 399.00 | 0.892 | 0.652 | 0.815 | 11.29 |
| Limestone | 4.76 | 285.31 | 0.632 | 0.449 | 0.572 | 6.08 |
| Shale (Fault) | 7.75 | 170.82 | 0.347 | 0.457 | 0.588 | 8.24 |

---

## ⚙️ Setup & Installation

### Prerequisites

```bash
Python >= 3.9
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```
numpy
pandas
scikit-learn
matplotlib
seaborn
torch          # or tensorflow / keras for CNN
jupyter
scipy
```

---

## 🚀 Running the Code

### 1. Generate Synthetic Data & Preprocess

```bash
python preprocessing/feature_engineering.py
```

This generates `data/synthetic_reservoir.csv` with physically consistent seismic attributes and porosity labels, with Gaussian noise added to simulate real-world uncertainty.

### 2. Train Random Forest

```bash
python models/random_forest.py
```

Trains a Random Forest regressor with GridSearchCV hyperparameter tuning and outputs feature importance scores.

### 3. Train 1D CNN

```bash
python models/cnn_1d.py
```

Trains the CNN model on the standardized seismic attribute vectors.

### 4. Run MC Dropout Inference

```bash
python models/mc_dropout.py
```

Runs stochastic forward passes through the trained CNN to generate mean predictions and uncertainty estimates (standard deviation).

### 5. Generate All Visualizations

```bash
python visualization/feature_importance.py
python visualization/correlation_heatmap.py
python visualization/attribute_vs_porosity.py
python visualization/mc_prediction_plot.py
```

### 6. Full Notebook Walkthrough

```bash
jupyter notebook notebooks/full_analysis.ipynb
```

---

## 📈 Figures Generated

| Figure | Description |
|---|---|
| Fig 1 | Prediction with Uncertainty (MC Dropout) — predicted porosity with confidence bands |
| Fig 2 | Feature Correlation Heatmap — pairwise correlation of all seismic attributes |
| Fig 3 | Seismic Attribute Importance — Random Forest feature importance ranking |
| Fig 4 | Frequency vs Porosity — scatter plot with trend |
| Fig 5 | RMS vs Porosity — scatter plot with trend |
| Fig 6 | Amplitude vs Porosity — scatter plot with trend |

---

## 🧠 Model Details

### Random Forest Regressor
- **Library:** `scikit-learn`
- **Tuning:** GridSearchCV over `n_estimators`, `max_depth`, `min_samples_split`
- **Key Output:** Feature importance array, R², RMSE
- **Advantage:** Interpretable, robust to noise, no scaling required

### 1D Convolutional Neural Network
- **Input:** 1D vector of 5 seismic attributes per sample
- **Architecture:** Conv1D → ReLU → Dropout → Dense → Output
- **Training:** Adam optimizer, MSE loss, Early Stopping
- **Key Output:** Porosity prediction per sample

### Monte Carlo Dropout
- Dropout layers remain **active during inference**
- Run **N forward passes** (e.g., N=100) per sample
- **Mean** of passes = final prediction
- **Std Dev** of passes = prediction uncertainty
- Enables confidence-aware outputs critical for drilling decisions

---

## 🏭 Real-World Significance

This framework addresses a critical challenge in the **oil & gas industry**: making reliable subsurface decisions under uncertainty.

- **Drilling Optimization:** Uncertainty estimates help geoscientists flag high-risk zones before costly drilling operations
- **Reservoir Development Planning:** Accurate porosity maps enable better estimation of hydrocarbon volumes
- **Interpretability for Geophysicists:** Random Forest importance scores translate ML predictions into domain-understandable language
- **Scalability:** The pipeline can be extended to real-field seismic data (3D seismic cubes) with minimal modification
- **Future Directions:** Physics-informed neural networks (PINNs), multi-modal learning integrating well-log and seismic data simultaneously

---

## 📚 References

1. Seismic Reservoir Modeling Python package — [SeReMpy GitHub](https://github.com/seismicreservoirmodeling/SeReMpy) (Accessed 2025)
2. W. Nie, J. Gu, B. Li, X. Wen, X. Nie — *210*, 112984 (2022)
3. L. Breiman — Random Forests, *Machine Learning*, 45, 5–32 (2001)
4. Y. Gal, Z. Ghahramani — Dropout as a Bayesian Approximation, *ICML*, 48, 1050–1059 (2016)
5. I. Goodfellow, Y. Bengio, A. Courville — *Deep Learning*, MIT Press (2016)
6. B. Russell — *Introduction to Seismic Inversion Methods*, SEG Books (2014)

## 📜 Citation

If you use this work, please cite:

```bibtex
@inproceedings{bhargava2026seismic,
  title     = {Comparative Analysis of Ensemble Learning and Deep Neural Networks
               for Seismic-Based Oil Reservoir Characterization},
  author    = {Bhargava, Devisha and Banerjee, Kaustav and Shaurya, Shivam},
  booktitle = {RGIPT Institute Day 2026 — Sustainable Energy Solutions},
  year      = {2026},
  month     = {February},
  address   = {Rajiv Gandhi Institute of Petroleum Technology, Jais}
}
```

---

*Department of Computer Science & Engineering | Department of Electrical & Electronics Engineering*  
*Rajiv Gandhi Institute of Petroleum Technology, Jais · February 2026*
