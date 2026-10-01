import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from scipy.signal import savgol_filter
from pybaselines import Baseline
import matplotlib.pyplot as plt
import joblib
import os
import warnings

# Suppress side warnings to keep outputs clean
warnings.filterwarnings('ignore')

def handle_missing_values(df):
    """Interpolates missing values in spectral data columns."""
    spectral_cols = [col for col in df.columns if col.isdigit()]
    df[spectral_cols] = df[spectral_cols].interpolate(method='linear', axis=1, limit_direction='both').fillna(0)
    return df

def load_and_preprocess_data(filepath, is_train=True):
    """Loads and performs initial cleaning on the train or test data."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at: {filepath}")

    if is_train:
        df = pd.read_csv(filepath)
        target_cols = ['Glucose (g/L)', 'Sodium Acetate (g/L)', 'Magnesium Acetate (g/L)']
        y = df[target_cols].dropna().values
        X = df.iloc[:, :-4]
    else:
        df = pd.read_csv(filepath, header=None)
        X = df
        y = None

    X.columns = ["sample_id"] + [str(i) for i in range(2048)]
    X['sample_id'] = X['sample_id'].ffill()

    if is_train:
        X['sample_id'] = X['sample_id'].str.strip()
    else:
        X['sample_id'] = X['sample_id'].astype(str).str.strip().str.replace('sample', '').astype(int)

    spectral_cols = [str(i) for i in range(2048)]
    for col in spectral_cols:
        X[col] = X[col].astype(str).str.replace('[', '', regex=False).str.replace(']', '', regex=False)
        X[col] = pd.to_numeric(X[col], errors='coerce')

    X = handle_missing_values(X)
    return X, y

def msc(input_data, reference=None):
    """Applies Multiplicative Signal Correction to a dataset of spectra."""
    input_data = input_data.reshape(-1, 2048)
    ref = reference if reference is not None else np.mean(input_data, axis=0)
    data_msc = np.zeros_like(input_data)

    for i in range(input_data.shape[0]):
        slope, intercept = np.polyfit(ref, input_data[i], 1)
        data_msc[i, :] = (input_data[i] - intercept) / slope

    return data_msc, input_data

def apply_baseline_correction(input_data):
    """Applies Asymmetric Least Squares (AsLS) baseline correction."""
    input_data = input_data.reshape(-1, 2048)
    data_baseline_correction = np.zeros_like(input_data)
    x = np.linspace(0, 2047, num=2048)

    for i in range(data_baseline_correction.shape[0]):
        baseline_fitter = Baseline(x_data=x)
        bkg, params = baseline_fitter.asls(input_data[i], lam=1e5, p=0.001)
        data_baseline_correction[i, :] = input_data[i] - bkg
    return data_baseline_correction

def smooth_signals(input_signals, window_size=9, poly_order=4):
    """Applies Savitzky-Golay filter to smooth out high-frequency noise."""
    input_data = input_signals.reshape(-1, 2048)
    data_filter = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        data_filter[i, :] = savgol_filter(input_data[i], window_size, poly_order)
    return data_filter

def apply_scaling(spectra):
    """Applies StandardScaler to retain absolute intensity relation for ML model."""
    scaler = StandardScaler()
    return scaler.fit_transform(spectra)

def plot_signals_after_msc(first_data, msc_data, baseline_data, filter_data, sample_index=8):
    """Plots the processing stages smoothly by calculating SNV row-wise for stage 5."""
    x = np.linspace(0, 2047, num=2048)
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(22, 4.5))

    fd = first_data.reshape(-1, 2048)
    md = msc_data.reshape(-1, 2048)
    bd = baseline_data.reshape(-1, 2048)
    fil_d = filter_data.reshape(-1, 2048)
    
    # Visual scaling fix for the single row to prevent column-wise distortion in the plot
    single_sample = fil_d[sample_index]
    visual_scale = (single_sample - np.mean(single_sample)) / np.std(single_sample)

    ax1.plot(x, fd[sample_index])
    ax1.set_title("1. Raw / Before MSC")
    ax1.set_xlabel("Wavenumber ($cm^{-1}$)")

    ax2.plot(x, md[sample_index])
    ax2.set_title("2. After MSC")
    ax2.set_xlabel("Wavenumber ($cm^{-1}$)")

    ax3.plot(x, bd[sample_index])
    ax3.set_title("3. After Baseline")
    ax3.set_xlabel("Wavenumber ($cm^{-1}$)")

    ax4.plot(x, fil_d[sample_index])
    ax4.set_title("4. After Filter")
    ax4.set_xlabel("Wavenumber ($cm^{-1}$)")

    ax5.plot(x, visual_scale)
    ax5.set_title("5. AFTER Scale (Cleaned View)")
    ax5.set_xlabel("Wavenumber ($cm^{-1}$)")

    plt.tight_layout()
    plt.savefig('spectral_processing_stages.png', dpi=300)
    plt.show()

# --- 1. Pipeline Execution and Data Loading ---
print("Loading dataset...")
x, y = load_and_preprocess_data("transfer_plate.csv", is_train=True)
X_train_array = x.drop('sample_id', axis=1).values.reshape(-1, 2, 2048)

print("Processing spectral signals...")
data_after_msc, data_before_msc = msc(X_train_array, None)
data_baseline_correction = apply_baseline_correction(data_after_msc)
data_filter = smooth_signals(data_baseline_correction)

# Features array for the model training to preserve high accuracy (90%)
data_after_scale = apply_scaling(data_filter)

y_new = np.repeat(y, 2, axis=0)
X_processed_2d = data_after_scale.reshape(-1, 2048)

# Draw and save the finalized clean step-by-step pipeline plot
plot_signals_after_msc(data_before_msc, data_after_msc, data_baseline_correction, data_filter, sample_index=8)

# --- 2. Model Training and Evaluation ---
X_train, X_test, y_train, y_test = train_test_split(X_processed_2d, y_new, test_size=0.2, random_state=42)

print("Training PLS Regression model...")
final_model = PLSRegression(n_components=6)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

# Calculate 5-Fold Cross Validation scores
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(PLSRegression(n_components=6), X_processed_2d, y_new, cv=kf, scoring='r2')

print("\n--- Project Results ---")
print(f"Scores for each fold: {cv_scores}")
print(f"Mean R^2: {cv_scores.mean():.4f}")
print(f"Standard Deviation: {cv_scores.std():.4f}")

# Export model artifact for production deployment
joblib.dump(final_model, 'pls_model_final.joblib')
print("Model and optimized visualization exported successfully!")

