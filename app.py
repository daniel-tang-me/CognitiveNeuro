"""Streamlit MVP: EEG Alzheimer's Disease research prototype."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import welch

MODEL_PATH = Path(__file__).with_name("alzheimer_rf_model.pkl")
FS = 250
EPOCH = 250
EPS = 1e-8

CHANNELS = [
    "Fp1","Fp2","F7","F3","Fz","F4","F8","T3",
    "C3","Cz","C4","T4","T5","P3","Pz","P4"
]

FEATURES = [
    f"{ch}_{feat}"
    for ch in CHANNELS
    for feat in ("std", "mean", "theta_alpha_ratio", "slowing_index")
]


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH.name} is missing. Put it beside app.py."
        )
    return joblib.load(MODEL_PATH)


def extract_features(df):
    """Exactly reproduce the 64 features used by the training pipeline."""
    rows = []
    for i in range(len(df) // EPOCH):
        x = df.iloc[i * EPOCH:(i + 1) * EPOCH]
        row = {}

        for ch in CHANNELS:
            signal = x[ch].to_numpy(dtype=float)
            row[f"{ch}_std"] = float(np.std(signal))
            row[f"{ch}_mean"] = float(np.mean(signal))

            freq, psd = welch(
                signal, fs=FS, nperseg=min(len(signal), 128)
            )
            delta = np.sum(psd[(freq >= .5) & (freq < 4)])
            theta = np.sum(psd[(freq >= 4) & (freq < 8)])
            alpha = np.sum(psd[(freq >= 8) & (freq < 13)])
            beta = np.sum(psd[(freq >= 13) & (freq < 30)])

            row[f"{ch}_theta_alpha_ratio"] = float(
                theta / (alpha + EPS)
            )
            row[f"{ch}_slowing_index"] = float(
                (delta + theta) / (alpha + beta + EPS)
            )

        rows.append(row)

    return pd.DataFrame(rows, columns=FEATURES)


def validate(df):
    errors, warnings = [], []
    missing = [c for c in CHANNELS if c not in df.columns]

    if missing:
        errors.append("Missing required EEG channel(s): " + ", ".join(missing))
        return False, errors, warnings

    if "status" in df.columns:
        warnings.append(
            "A 'status' column was found and will be ignored. "
            "Predictions do not require a diagnosis label."
        )

    eeg = df[CHANNELS].copy()
    for ch in CHANNELS:
        vals = pd.to_numeric(eeg[ch], errors="coerce")
        bad = int(vals.isna().sum())
        if bad:
            errors.append(f"{ch} contains {bad:,} missing/non-numeric value(s).")

    if errors:
        return False, errors, warnings

    eeg = eeg.astype(float)
    if not np.isfinite(eeg.to_numpy()).all():
        errors.append("The EEG data contains infinite values.")
        return False, errors, warnings

    if len(eeg) < EPOCH:
        errors.append("At least 250 samples are required (1 second at 250 Hz).")
        return False, errors, warnings

    remainder = len(eeg) % EPOCH
    if remainder:
        warnings.append(
            f"The final {remainder} sample(s) are incomplete and will be ignored."
        )

    std = eeg.std()
    flat = std[std == 0].index.tolist()
    if flat:
        warnings.append(
            "Completely flat channel(s) detected: " + ", ".join(flat)
        )

    return True, errors, warnings


def predict(model, features):
    pred = model.predict(features)
    proba = model.predict_proba(features)

    classes = {int(c): i for i, c in enumerate(model.classes_)}
    if 0 not in classes or 1 not in classes:
        raise ValueError(
            "The loaded model is not the expected binary status 0/1 model."
        )

    ad = proba[:, classes[1]]
    mean_ad = float(ad.mean())
    ad_fraction = float(np.mean(pred == 1))

    return {
        "label": int(mean_ad >= .5),
        "score": mean_ad,
        "ad_fraction": ad_fraction,
        "epochs": len(features),
        "ad_epochs": int(np.sum(pred == 1)),
    }


def instructions():
    with st.expander("📄 CSV requirements", expanded=True):
        st.markdown(
            """
**Required EEG channels**

`Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4`

**Input assumptions**
- Sampling rate: **250 Hz**
- One row = one EEG sample
- At least **250 rows**
- Numeric EEG values only
- No missing/NaN EEG values
- Complete 250-sample windows are analyzed as 1-second epochs
- A `status` column is **not required** and is ignored if present
- Extra columns are allowed

The app uses the same 16-channel, 250 Hz, 1-second/64-feature pipeline used during training.
"""
        )
        template = pd.DataFrame({c: [0.0] for c in CHANNELS})
        st.download_button(
            "Download CSV column template",
            template.to_csv(index=False),
            "eeg_upload_template.csv",
            "text/csv",
            use_container_width=True,
        )


def limitations():
    with st.expander("⚠️ Accuracy, fairness & edge cases"):
        st.markdown(
            """
**Accuracy.** The original held-out evaluation achieved **70.6% accuracy**.
The AD-associated class had **92% precision** and **53% recall**. A separate
recording-group investigation gave performance in roughly the mid-70% range,
but results varied substantially between recordings.

**Fairness.** The source data does not contain enough demographic/subject
metadata to measure performance across age, sex, ethnicity, or other groups.
This MVP therefore makes **no claim of demographic fairness**.

**Edge cases.** The app rejects missing channels, missing/non-numeric values,
infinite values, and files shorter than one complete epoch. It warns about
incomplete final epochs and completely flat channels.

**Clinical limitation.** The source dataset has no explicit patient/subject
IDs, so evaluation cannot establish true independent-patient generalization.
This is a research/demo prototype, **not a medical diagnostic system**.
"""
        )


def main():
    st.set_page_config(
        page_title="EEG Alzheimer's Research Prototype",
        page_icon="🧠",
        layout="wide",
    )

    st.title("🧠 EEG Alzheimer's Disease Research Prototype")
    st.caption("Upload EEG → validate → extract qEEG features → analyze")

    st.warning(
        "Research demonstration only. This tool does not diagnose Alzheimer's "
        "disease and should not be used to make medical decisions."
    )

    instructions()
    st.divider()

    try:
        model = load_model()
    except Exception as e:
        st.error(f"Unable to load the trained model: {e}")
        st.stop()

    st.subheader("1. Upload EEG data")
    uploaded = st.file_uploader(
        "Choose a CSV containing the 16 required EEG channels",
        type=["csv"],
        help="Expected raw EEG samples at 250 Hz.",
    )

    if uploaded is None:
        st.info("Upload a compatible CSV to begin.")
        limitations()
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read this CSV: {e}")
        return

    st.subheader("2. Validate input")
    ok, errors, warnings = validate(df)

    if errors:
        st.error("This file cannot be analyzed yet.")
        for e in errors:
            st.write(f"• {e}")
        return

    for w in warnings:
        st.warning(w)

    n = len(df)
    epochs = n // EPOCH
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Samples", f"{n:,}")
    c2.metric("Duration", f"{n / FS:.1f} s")
    c3.metric("Complete epochs", f"{epochs:,}")
    c4.metric("Channels", "16 / 16")

    st.success("Input format looks valid.")

    with st.expander("Preview uploaded EEG"):
        st.dataframe(df[CHANNELS].head(10), use_container_width=True)

    st.subheader("3. Analyze recording")

    if st.button("🧠 Analyze EEG", type="primary", use_container_width=True):
        progress = st.progress(0, "Preparing EEG data...")

        try:
            eeg = df[CHANNELS].astype(float)

            progress.progress(20, "Extracting 64 qEEG features...")
            features = extract_features(eeg)

            if features.empty:
                raise ValueError("No complete 1-second epochs were extracted.")

            if list(features.columns) != FEATURES:
                raise ValueError("Generated features do not match model schema.")

            progress.progress(65, "Running the trained Random Forest...")
            result = predict(model, features)

            progress.progress(100, "Analysis complete.")
            progress.empty()

            st.subheader("Analysis result")
            if result["label"] == 1:
                st.error("Model output: AD-associated pattern detected")
            else:
                st.success("Model output: non-AD-associated pattern detected")

            st.metric(
                "AD-associated model score",
                f"{result['score'] * 100:.1f}%",
                help=(
                    "Average Random Forest probability for class 1 across "
                    "the recording. This is NOT a clinical probability."
                ),
            )

            a, b, c = st.columns(3)
            a.metric("Valid 1-second epochs", f"{result['epochs']:,}")
            b.metric("AD-associated epochs", f"{result['ad_epochs']:,}")
            c.metric(
                "AD-associated epoch fraction",
                f"{result['ad_fraction'] * 100:.1f}%",
            )

            with st.expander("How to interpret this"):
                st.write(
                    "The model scores each 1-second EEG window. The app averages "
                    "those probabilities into one recording-level model score."
                )
                st.warning(
                    "This score is not the probability that a person has "
                    "Alzheimer's disease. It is a research-model output."
                )

            st.caption(
                f"Processed {result['epochs']:,} complete epochs. "
                "The upload is analyzed for this session and is not used "
                "to retrain the model."
            )

        except Exception as e:
            progress.empty()
            st.error(f"Analysis failed: {e}")

    limitations()


if __name__ == "__main__":
    main()
