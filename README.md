# CognitiveNeuro

A Streamlit computational EEG research prototype for experimental pattern
analysis. The application loads a serialized scikit-learn Random Forest pipeline,
extracts 64 qEEG features from 16-channel raw recordings, and reports model class
similarity in a non-clinical research interface.

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The CSV workflow expects at least 250 rows sampled at 250 Hz and these channels:
`Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4`.

This software is for research demonstration only. It is not a medical device and
does not provide a diagnosis or individual clinical risk estimate.
