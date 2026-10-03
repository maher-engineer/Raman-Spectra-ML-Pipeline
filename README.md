# Raman Spectroscopy Signal Processing & ML Pipeline

## 📌 Project Overview
This repository contains an end-to-end Machine Learning and Chemometrics pipeline designed to process and predict multi-output chemical concentrations from raw **Raman Spectroscopy data**. 

Coming from a **Mechatronics Engineering background**, I designed this architecture to interface directly with raw spectral sensor streams, successfully overcoming classic industrial signal challenges like intense fluorescence background noise and light scattering.

## 🛠️ Tech Stack & Libraries
- **Environment:** Jupyter Notebook (`.ipynb`)
- **Language:** Python
- **Signal Processing:** `pybaselines` (Asymmetric Least Squares), `scipy` (Savitzky-Golay)
- **Machine Learning:** `scikit-learn` (Partial Least Squares Regression - PLSR)
- **Data Engineering:** `pandas`, `numpy`
- **Visualization:** `matplotlib`

## 🚀 Pipeline Architecture & Stages
The core strength of this pipeline lies in transforming raw, noisy sensor data into clean, standardized features optimized for machine learning. The entire sequence is executed and visualized inside the notebook:
1. **Linear Imputation:** Dynamically handles missing spectral lines.
2. **Multiplicative Signal Correction (MSC):** Eliminates variations caused by physical light scattering across samples.
3. **Automated Baseline Subtraction:** Employs Asymmetric Least Squares (AsLS) smoothing to suppress background fluorescence without destroying chemical peak properties.
4. **Signal Smoothing:** Uses a Savitzky-Golay filter to suppress high-frequency instrumental and electronic noise.
5. **Chemometric Predictive Modeling:** Trains a **PLSRegression** model optimized for multi-output target prediction.

## 📊 Model Performance & Validation
To guarantee the model's reliability and safeguard against data leakage, the pipeline undergoes a strict **5-Fold Cross-Validation** sequence.

The trained **PLSRegression (n_components=6)** model achieved highly robust and stable results:
- **Fold-by-Fold R² Scores:** `[0.8998, 0.8979, 0.9161, 0.8891, 0.8653]`
- **Mean R² Score:** **`0.8937`** (89.37% predictive accuracy)
- **Standard Deviation:** **`0.0166`** (Demonstrating excellent model stability)

*Note: All step-by-step spectral plots and the Actual vs. Predicted scatter plot are rendered natively inside the notebook.*

## 📂 Project Structure
- `Raman_Spectroscopy_Pipeline.ipynb`: The complete interactive notebook containing the data loading, signal cleaning, modeling, and native visualizations.
- `pls_modelii.joblib`: The production-ready serialized model artifact.
