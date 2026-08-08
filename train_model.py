# train_model.py
# Research prototype only: this model supports EEG-based decision research and is
# not intended to diagnose Alzheimer's disease or replace clinical assessment.

import pandas as pd
import re
import warnings

import joblib
import numpy as np
from scipy.signal import welch
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

DATA_FILE = "eeg_data.csv"
MODEL_FILE = "alzheimer_rf_model.pkl"
EPSILON = 1e-8


def normalize_name(name):
    """Normalize column names so common naming variations can be matched."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_matching_column(columns, feature_name):
    """
    Find a spectral-power column using common variants such as:
    Theta, theta_power, Theta Power, theta_band_power, etc.
    """
    normalized_columns = {normalize_name(col): col for col in columns}
    feature_key = normalize_name(feature_name)

    preferred_names = [
        feature_key,
        f"{feature_key}power",
        f"{feature_key}band",
        f"{feature_key}bandpower",
        f"{feature_key}absolutepower",
        f"{feature_key}relativepower",
        f"{feature_key}powermean",
    ]

    for name in preferred_names:
        if name in normalized_columns:
            return normalized_columns[name]

    for normalized, original in normalized_columns.items():
        if feature_key in normalized and "ratio" not in normalized:
            return original

    return None


def choose_target_column(df):
    """Identify the most likely cognitive-state label column."""
    target_priority = [
        "neurocognitive_state",
        "neurocognitivestate",
        "status",
        "label",
        "class",
        "state",
        "diagnosis",
        "cognitive_state",
        "cognitivestate",
        "condition",
        "group",
        "target",
    ]

    normalized_map = {normalize_name(col): col for col in df.columns}
    possible_targets = []

    for candidate in target_priority:
        if candidate in normalized_map:
            possible_targets.append(normalized_map[candidate])

    # Include object/category columns as additional possible label candidates.
    for column in df.columns:
        if (
            (df[column].dtype == "object" or str(df[column].dtype) == "category")
            and column not in possible_targets
        ):
            possible_targets.append(column)

    if not possible_targets:
        raise ValueError(
            "No likely target column was found. Expected a column such as "
            "status, label, class, state, or Neurocognitive_State."
        )

    print("\nPossible target columns:")
    for column in possible_targets:
        print(f"\n{column}:")
        print(df[column].dropna().unique())

    selected_target = possible_targets[0]
    print(f"\nSelected target column: {selected_target}")
    return selected_target


def create_qeeg_features(df, spectral_columns):
    """Create qEEG relative power features and slowing ratios."""
    engineered = df.copy()

    delta_col = spectral_columns.get("Delta")
    theta_col = spectral_columns.get("Theta")
    alpha_col = spectral_columns.get("Alpha")
    beta_col = spectral_columns.get("Beta")
    gamma_col = spectral_columns.get("Gamma")

    # Ensure spectral columns are numeric
    for band_col in [delta_col, theta_col, alpha_col, beta_col, gamma_col]:
        if band_col and band_col in engineered.columns:
            engineered[band_col] = pd.to_numeric(engineered[band_col], errors="coerce")

    # 1. Calculate Total Power across available frequency bands
    band_cols = [c for c in [delta_col, theta_col, alpha_col, beta_col, gamma_col] if c]
    engineered["Total_Power"] = engineered[band_cols].sum(axis=1)

    # 2. Compute Relative Band Powers (Normalized values between 0 and 1)
    for band_name, col_name in spectral_columns.items():
        if col_name:
            engineered[f"Relative_{band_name}"] = engineered[col_name] / (
                engineered["Total_Power"] + EPSILON
            )

    # 3. Compute Key Biomarker Ratios
    if theta_col and alpha_col:
        engineered["Theta_Alpha_Ratio"] = engineered[theta_col] / (
            engineered[alpha_col] + EPSILON
        )

    if theta_col and beta_col:
        engineered["Theta_Beta_Ratio"] = engineered[theta_col] / (
            engineered[beta_col] + EPSILON
        )

    if delta_col and alpha_col:
        engineered["Delta_Alpha_Ratio"] = engineered[delta_col] / (
            engineered[alpha_col] + EPSILON
        )

    engineered = engineered.replace([np.inf, -np.inf], np.nan)
    return engineered

def extract_epoch_features(df, window_size=250, sampling_rate=250):
    """
    Groups continuous EEG time-series into 1-second epoch windows (250 samples)
    and computes time-domain stats + Welch Power Spectral Density features per channel.
    """
    eeg_channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4']
    
    # Assign an epoch index to every window_size chunk of rows
    df_work = df.copy()
    df_work['epoch_id'] = np.arange(len(df_work)) // window_size

    # Filter out partial remaining trailing epoch at the end
    valid_epochs = df_work['epoch_id'].value_counts()
    valid_epoch_ids = valid_epochs[valid_epochs == window_size].index
    df_work = df_work[df_work['epoch_id'].isin(valid_epoch_ids)]

    print(f"\nProcessing {len(valid_epoch_ids)} epoch windows ({window_size} samples / 1 sec each)...")

    feature_rows = []
    
    # Process each epoch chunk efficiently
    for epoch_id, group in df_work.groupby('epoch_id'):
        row_feat = {}
        
        # Target status label for this epoch window
        status_label = group['status'].iloc[0]
        row_feat['status'] = status_label
        
        for col in eeg_channels:
            signal = group[col].values
            
            # 1. Time-Domain Features
            row_feat[f'{col}_std'] = np.std(signal)
            row_feat[f'{col}_mean'] = np.mean(signal)
            
            # 2. Spectral Power Density via Welch Method
            freqs, psd = welch(signal, fs=sampling_rate, nperseg=min(len(signal), 128))
            
            delta = np.sum(psd[(freqs >= 0.5) & (freqs < 4)])
            theta = np.sum(psd[(freqs >= 4) & (freqs < 8)])
            alpha = np.sum(psd[(freqs >= 8) & (freqs < 13)])
            beta  = np.sum(psd[(freqs >= 13) & (freqs < 30)])
            
            # 3. Frequency Ratios & Slowing Markers
            row_feat[f'{col}_theta_alpha_ratio'] = theta / (alpha + EPSILON)
            row_feat[f'{col}_slowing_index'] = (delta + theta) / (alpha + beta + EPSILON)
            
        feature_rows.append(row_feat)

    df_features = pd.DataFrame(feature_rows)
    return df_features

def main():
    warnings.filterwarnings("ignore", category=UserWarning)

    # Load preprocessed EEG spectral-feature dataset.
    df = pd.read_csv("AD_all_patients.csv")

    print("All Columns in CSV:", df.columns.tolist())

    print("\n--- Missing Values Per Column ---")
    print(df.isnull().sum())

    print("\n--- Total Missing Values in Entire File ---")
    print(f"Total NaNs: {df.isnull().sum().sum()}")

    # Dataset inspection.
    print("Dataset shape:", df.shape)
    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())

    target_column = choose_target_column(df)

# Transform raw time series into windowed 1-second qEEG epoch features
    df_epochs = extract_epoch_features(df, window_size=250, sampling_rate=250)

    selected_features = [col for col in df_epochs.columns if col != 'status']

    print(f"\nExtracted {len(selected_features)} engineered qEEG features across epochs.")

    X = df_epochs[selected_features]
    y = df_epochs['status']

    # --- Target Label Distribution Check ---
    print("\n--- Target Label Distribution Check ---")
    print(y.value_counts(dropna=False))
    print("Unique Classes:", y.unique())
    print("---------------------------------------\n")

    # --- ADD DEBUG PRINTS HERE ---
    print("\n--- Model Input Data Preview ---")
    print(X.head())
    print("\nFeature Columns Being Used:", X.columns.tolist())
    # -----------------------------

    # Remove samples lacking a class label; feature missing values stay in the
    # pipeline and are imputed only after the train/test split.
    valid_target_mask = y.notna()
    X = X.loc[valid_target_mask]
    y = y.loc[valid_target_mask]

    if y.nunique() < 2:
        raise ValueError("The target column must contain at least two classes.")

    class_counts = y.value_counts()
    if class_counts.min() < 2:
        raise ValueError(
            "Each class needs at least two samples to use a stratified train/test split."
        )

    # ------------------------------------------------------------------
    # LARGE-BLOCK SESSION GROUPING (5-minute continuous session holdouts)
    # Groups every 300 continuous 1-second epochs into a session block
    # so whole multi-minute recordings are kept together in train or test.
    # ------------------------------------------------------------------
    BLOCK_SIZE = 300  # 300 epochs = 5 minutes of continuous signal
    df_epochs['large_block_id'] = np.arange(len(df_epochs)) // BLOCK_SIZE

    print(f"\nGrouped data into {df_epochs['large_block_id'].nunique()} large session blocks.")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=df_epochs['large_block_id']))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(f"\nTrain set: {len(X_train)} epochs | Test set: {len(X_test)} epochs")
    print("Test set target label distribution:\n", y_test.value_counts())

    # The complete preprocessing and model workflow is saved together.
    # Random Forest is a low-compute, robust choice for small tabular qEEG datasets.
    # max_depth constrains tree complexity to help reduce overfitting.

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric_features",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),  # Normalizes variance across features
                    ]
                ),
                selected_features,
            )
        ],
        remainder="drop",
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                    
                ),
            ),
        ]
    )

    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)

    print("\nEvaluation Results")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print(
        "Precision:",
        round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
    )
    print(
        "Recall:",
        round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
    )
    print(
        "F1-score:",
        round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
    )

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # This saved pipeline supports model.predict() and model.predict_proba().
    # For Streamlit, pass a pandas DataFrame containing the selected feature
    # columns printed above, including available engineered ratio columns.
    joblib.dump(model_pipeline, MODEL_FILE)

    print(f"\nModel saved successfully as: {MODEL_FILE}")
    print("\nImportant limitation: small EEG datasets can produce optimistic and")
    print("unstable results. Validate on independent subjects and avoid using this")
    print("research prototype as a medical diagnosis system.")


if __name__ == "__main__":
    main()