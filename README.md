# Raman-Spectra-ML-Pipeline
End-to-end Python pipeline using Machine Learning to clean, filter, and predict quantitative values from raw Raman spectroscopy data.
# Raman Spectroscopy Signal Processing & ML Pipeline

## 📌 Project Overview
This repository contains an end-to-end Machine Learning and Chemometrics pipeline designed for processing and predicting chemical concentrations from **Raman Spectroscopy data**. 

Coming from a **Mechatronics Engineering background**, I built this architecture to interface with physical spectral sensors, successfully overcoming industrial data challenges such as intense fluorescence background noise and signal scattering.

## 🛠️ Tech Stack & Libraries
- **Language:** Python
- **Signal Processing:** `pybaselines` (Asymmetric Least Squares), `scipy` (Savitzky-Golay)
- **Machine Learning:** `scikit-learn` (Partial Least Squares Regression - PLSR)
- **Data Engineering:** `pandas`, `numpy`
- **Visualization:** `matplotlib`

## 🚀 Pipeline Architecture
The codebase processes 2,048-dimensional spectral features through the following sequential modules:
1. **Linear Imputation:** Dynamically handles missing spectral lines.
2. **Multiplicative Signal Correction (MSC):** Eliminates variations caused by physical light scattering.
3. **Automated Baseline Subtraction:** Employs Asymmetric Least Squares (AsLS) smoothing to suppress background fluorescence.
4. **Signal Smoothing:** Uses a Savitzky-Golay filter to suppress high-frequency instrumental noise while preserving peak shapes.
5. **Chemometric Predictive Modeling:** Trains a **PLSRegression** model optimized for multi-output target prediction.

## 📊 Model Evaluation & Results
The pipeline utilizes a **5-Fold Cross-Validation** strategy to ensure model robustness and eliminate data leakage. Below is the performance evaluation showing the correlation between the actual and predicted concentrations:

![Actual vs Predicted](./results.png)

*Figure: Evaluation plot demonstrating the predictive accuracy of the PLS Regression model against the target concentrations.*

## 📂 Project Structure
- `main.py`: The core executable Python pipeline script.
- `requirements.txt`: Environment dependencies.
<img width="1539" height="449" alt="Image" src="https://github.com/user-attachments/assets/62590906-249b-4838-8950-866d3b44b556" />
- `pls_model.joblib`: The production-ready serialized model artifact.
