from datetime import datetime

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from assistant_component import render_analysis_assistant
from model_service import (
    BANDS,
    CHANNELS,
    EPOCH,
    FEATURES,
    FS,
    extract_features,
    load_model,
    predict,
    project_manual_bands,
    validate_raw_eeg,
)
from ui_components import band_chart, disclaimer, inject_styles, page_intro, research_probability


PAGES = [
    "Overview",
    "EEG Analysis",
    "Alzheimer's Disease",
    "About the Model",
]
NAV_LABELS = {
    "Overview": "Overview",
    "EEG Analysis": "EEG Analysis",
    "Alzheimer's Disease": "Alzheimer's",
    "About the Model": "Model",
}


def initialize_state():
    defaults = {
        "page": "Overview",
        "analysis_result": None,
        "analysis_bands": None,
        "analysis_source": None,
        "analysis_timestamp": None,
        "analysis_context": None,
        "analysis_recommendation": None,
        "messages": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def navigate(page: str):
    st.session_state.page = page
    st.session_state.top_navigation = NAV_LABELS.get(page, "Overview")


def product_navigation():
    st.markdown(
        '<div class="product-header"><div><div class="product-name">CognitiveNeuro</div>'
        '<div class="product-descriptor">EEG Computational Analysis</div></div>'
        '<div class="product-descriptor">Research / Educational Prototype</div></div>',
        unsafe_allow_html=True,
    )
    labels = list(NAV_LABELS.values())
    current_label = NAV_LABELS.get(st.session_state.page, "Overview")
    navigation_default = {} if "top_navigation" in st.session_state else {"default": current_label}
    selected = st.segmented_control(
        "Primary navigation",
        labels,
        label_visibility="collapsed",
        key="top_navigation",
        **navigation_default,
    )
    page_lookup = {label: page for page, label in NAV_LABELS.items()}
    if selected and page_lookup[selected] != st.session_state.page:
        st.session_state.page = page_lookup[selected]
        st.rerun()


def overview_page():
    left, right = st.columns([0.9, 1.1], gap="large", vertical_alignment="center")
    with left:
        st.markdown('<div class="eyebrow">Computational EEG Research Platform</div>', unsafe_allow_html=True)
        st.title("CognitiveNeuro")
        st.markdown(
            '<div class="lede">An experimental platform for exploring EEG-derived spectral '
            'features and machine-learning pattern analysis in cognitive neuroscience.</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        st.button("Begin EEG Analysis", type="primary", on_click=navigate, args=("EEG Analysis",))
    with right:
        st.markdown(
            '<div class="overview-aside"><div class="eyebrow">Analysis scope</div>'
            '<strong>Five spectral bands</strong><br>Manual exploratory input or validated '
            '16-channel EEG recordings.<br><br><strong>One experimental output</strong><br>'
            'Random Forest class-pattern similarity, presented with explicit research limitations.'
            '<svg class="overview-trace" viewBox="0 0 520 48" preserveAspectRatio="none" aria-hidden="true">'
            '<path d="M0 25 L45 25 L58 20 L66 32 L76 9 L88 38 L99 24 L145 24 L158 18 L168 29 '
            'L180 13 L192 35 L205 24 L258 24 L272 20 L281 28 L291 16 L303 32 L317 24 L365 24 '
            'L378 21 L388 27 L399 11 L411 37 L424 24 L520 24" fill="none" stroke="#6f94aa" '
            'stroke-width="1.4" vector-effect="non-scaling-stroke"/></svg></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="editorial-rule">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Research workflow</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="workflow"><div class="workflow-step">EEG Features</div><span class="workflow-arrow">→</span>'
        '<div class="workflow-step">Experimental ML Analysis</div><span class="workflow-arrow">→</span>'
        '<div class="workflow-step">Pattern Analysis</div><span class="workflow-arrow">→</span>'
        '<div class="workflow-step">Interpretation</div></div>',
        unsafe_allow_html=True,
    )
    disclaimer()


def band_inputs() -> dict[str, float]:
    defaults = {"Delta": 1.0, "Theta": 1.0, "Alpha": 1.0, "Beta": 1.0, "Gamma": 1.0}
    columns = st.columns(5)
    values = {}
    for column, band in zip(columns, BANDS):
        with column:
            values[band] = st.number_input(
                band,
                min_value=0.0,
                value=defaults[band],
                step=0.1,
                format="%.3f",
                key=f"manual_{band.lower()}",
            )
    return values


def estimate_bands_from_raw(eeg: pd.DataFrame) -> dict[str, float]:
    """Provide a compact display profile; model inference still uses 64 features."""
    from scipy.signal import welch
    import numpy as np

    ranges = {"Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 13), "Beta": (13, 30), "Gamma": (30, 45)}
    signals = eeg[CHANNELS].to_numpy(dtype=float).T
    frequency, psd = welch(signals, fs=FS, nperseg=128, axis=-1)
    return {
        band: float(np.sum(psd[:, (frequency >= low) & (frequency < high)], axis=1).mean())
        for band, (low, high) in ranges.items()
    }


def render_results():
    result = st.session_state.analysis_result
    if not result:
        return
    st.markdown('<div id="analysis-results-anchor"></div>', unsafe_allow_html=True)
    if st.session_state.pop("scroll_to_analysis_results", False):
        components.html(
            """
            <script>
            const target = window.parent.document.getElementById('analysis-results-anchor');
            if (target) {
              window.setTimeout(() => target.scrollIntoView({behavior: 'smooth', block: 'start'}), 80);
            }
            </script>
            """,
            height=0,
            scrolling=False,
        )
    percentage = result["pattern_probability"] * 100
    ad_pattern_detected = result["pattern_probability"] >= 0.5
    final_classification = (
        "Alzheimer's-associated EEG pattern detected"
        if ad_pattern_detected
        else "No Alzheimer's-associated EEG pattern detected"
    )

    # Put the recording-level conclusion first so it cannot be missed below the
    # supporting metrics and technical details.
    st.markdown('<div class="eyebrow">Final analysis</div>', unsafe_allow_html=True)
    st.header(final_classification)
    if ad_pattern_detected:
        st.error(
            f"Model classification: ALZHEIMER'S-ASSOCIATED PATTERN — {percentage:.1f}% score"
        )
    else:
        st.success(
            f"Model classification: NON-ALZHEIMER'S-ASSOCIATED PATTERN — {percentage:.1f}% score"
        )
    st.write(
        "This is the final classification produced by the EEG model for this recording. "
        "It is not a confirmed medical diagnosis; Alzheimer's disease must be diagnosed "
        "by a qualified clinician using a complete clinical assessment."
    )

    st.markdown(
        '<div class="result-hero"><div class="eyebrow">Experimental ML Analysis</div>'
        f'<div class="result-value">{percentage:.1f}%</div>'
        '<div class="result-name">Alzheimer\'s Pattern Similarity</div>'
        '<div class="result-caption">Model-estimated classification probability based '
        'on the supplied EEG feature vector.</div></div>',
        unsafe_allow_html=True,
    )
    research_probability(result["pattern_probability"])
    st.caption(
        "A continuous experimental model output. The scale does not represent a clinical "
        "threshold, diagnosis, or individual risk category."
    )

    recording = (st.session_state.analysis_context or {}).get("recording")
    if recording:
        st.markdown('<div class="result-section"></div>', unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">Recording summary</div>', unsafe_allow_html=True)
        summary_columns = st.columns(4)
        summary_columns[0].metric("Samples", f"{recording['samples']:,}")
        summary_columns[1].metric("Duration", f"{recording['duration_seconds']:.1f} s")
        summary_columns[2].metric("Complete epochs", f"{recording['complete_epochs']:,}")
        summary_columns[3].metric("Channels", f"{recording['channels']} / {len(CHANNELS)}")

    output_label = final_classification
    st.markdown('<div class="result-section"></div>', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Analysis result</div>', unsafe_allow_html=True)
    st.subheader("Experimental Random Forest verdict")
    st.markdown(f"**Model output: {output_label}**")
    st.write(
        f"The supplied EEG input has **{result['pattern_probability'] * 100:.1f}% "
        "similarity to the AD-associated patterns learned by this Random Forest model**. "
        "This is an experimental classification verdict, not a medical diagnosis of "
        "Alzheimer's disease."
    )
    result_columns = st.columns(4)
    result_columns[0].metric(
        "AD-associated model score",
        f"{result['pattern_probability'] * 100:.1f}%",
    )
    result_columns[1].metric("Valid 1-second epochs", f"{result['records']:,}")
    result_columns[2].metric("AD-associated epochs", f"{result['pattern_records']:,}")
    result_columns[3].metric(
        "AD-associated epoch fraction",
        f"{result['pattern_records'] / result['records'] * 100:.1f}%",
    )

    chart_col, detail_col = st.columns([1.2, 0.8], gap="large")
    with chart_col:
        st.markdown('<div class="result-section"></div>', unsafe_allow_html=True)
        st.subheader("EEG Frequency Profile")
        band_chart(st.session_state.analysis_bands)
    with detail_col:
        st.markdown('<div class="result-section"></div>', unsafe_allow_html=True)
        st.subheader("Feature Details")
        band_frame = pd.DataFrame(
            {
                "Band": list(st.session_state.analysis_bands),
                "Input Value": list(st.session_state.analysis_bands.values()),
            }
        )
        st.dataframe(band_frame, hide_index=True, width="stretch")

    model = load_model()
    model_features = list(getattr(model, "feature_names_in_", []))
    model_classes = list(getattr(model, "classes_", []))
    display_feature_count = 5 if st.session_state.analysis_source.startswith("Manual") else len(model_features)
    st.markdown(
        '<div class="metadata-grid" style="margin:1.5rem 0">'
        f'<div class="metadata-row"><div class="metadata-label">Source</div><div class="metadata-value">{st.session_state.analysis_source}</div></div>'
        '<div class="metadata-row"><div class="metadata-label">Model</div><div class="metadata-value">Random Forest</div></div>'
        f'<div class="metadata-row"><div class="metadata-label">Displayed features</div><div class="metadata-value">{display_feature_count}</div></div>'
        '<div class="metadata-row"><div class="metadata-label">Status</div><div class="metadata-value">Validated</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Analysis details"):
        detail_left, detail_right = st.columns(2)
        detail_left.markdown("**Model type**  \nRandomForestClassifier")
        detail_left.markdown(f"**Number of features**  \n{len(model_features) or 64}")
        detail_left.markdown(
            f"**Prediction classes**  \n{', '.join(str(value) for value in model_classes) if model_classes else 'Unavailable'}"
        )
        detail_right.markdown(f"**Data source**  \n{st.session_state.analysis_source}")
        detail_right.markdown(f"**Timestamp**  \n{st.session_state.analysis_timestamp}")
        detail_right.markdown("**Model file status**  \nLoaded and available")
        st.markdown("**Feature order**")
        st.code(" → ".join(model_features) if model_features else "Feature metadata unavailable", language=None)

    st.markdown('<div class="result-section"></div>', unsafe_allow_html=True)
    st.subheader("Understanding this result")
    st.write(
        "This score is the Random Forest model's classification output for the supplied "
        "EEG feature vector, averaged across valid records when a CSV is analyzed. It "
        "describes similarity to a learned model class and does not establish whether "
        "someone has Alzheimer's disease."
    )
    st.caption("Use the floating CognitiveNeuro Assistant to discuss this active analysis without leaving the workspace.")

    if recording:
        recommendation = st.session_state.get("analysis_recommendation")
        if recommendation:
            st.markdown('<div class="result-section"></div>', unsafe_allow_html=True)
            st.subheader("Suggested next steps")
            st.write(recommendation)
            st.caption(
                "For additional follow-up questions, open the circular CognitiveNeuro "
                "Assistant in the bottom-right. It already has this analysis context."
            )

        with st.expander("Accuracy, fairness & edge cases"):
            st.markdown(
                """
                **Accuracy.** The project-reported original held-out evaluation achieved
                **70.6% accuracy**. The AD-associated class had **92% precision** and
                **53% recall**. A separate recording-group investigation reportedly produced
                performance in approximately the mid-70% range, with substantial variation
                between recordings. These figures are historical project results and are not
                independent clinical validation.

                **Fairness.** The source data does not contain enough demographic or subject
                metadata to measure performance across age, sex, ethnicity, or other groups.
                This prototype therefore makes **no claim of demographic fairness**.

                **Edge cases.** The application rejects missing channels, missing or
                non-numeric values, infinite values, and files shorter than one complete epoch.
                It warns about incomplete final epochs and completely flat channels.

                **Clinical limitation.** The source dataset has no explicit patient or subject
                IDs, so evaluation cannot establish true independent-patient generalization.
                This is a research and educational prototype, **not a medical diagnostic system**.
                """
            )

        st.caption(
            f"Processed {result['records']:,} complete epochs. The upload is analyzed for "
            "this session and is not used to retrain the model."
        )
    disclaimer()


def save_analysis(
    result: dict,
    bands: dict[str, float],
    source: str,
    recording: dict | None = None,
    feature_values: dict[str, float] | None = None,
):
    """Persist one complete analysis so results and assistant share the same context."""
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    st.session_state.analysis_result = result
    st.session_state.analysis_bands = dict(bands)
    st.session_state.analysis_source = source
    st.session_state.analysis_timestamp = timestamp
    st.session_state.analysis_context = {
        "result": dict(result),
        "bands": dict(bands),
        "source": source,
        "timestamp": timestamp,
        "recording": recording,
        "feature_values": feature_values or dict(bands),
        "metadata": {
            "model_type": "RandomForestClassifier",
            "model_role": "sole numerical prediction source",
            "feature_count": len(feature_values or bands),
            "validation_status": "input validated",
            "clinical_validation": "not clinically validated",
        },
        "verdict": (
            "Alzheimer's-associated EEG pattern detected"
            if result["pattern_probability"] >= 0.5
            else "No Alzheimer's-associated EEG pattern detected"
        ),
    }
    st.session_state.analysis_recommendation = None
    st.session_state.scroll_to_analysis_results = True


def analysis_page():
    page_intro(
        "Scientific workspace",
        "EEG Analysis",
        "Spectral feature input and experimental model inference.",
    )

    # Always show a completed analysis above the input controls. Previously the
    # result was rendered after the CSV preview, which made it look as though no
    # result had been produced on long recordings.
    render_results()

    mode = st.segmented_control(
        "Input mode",
        ["Manual Input", "CSV Upload"],
        default="Manual Input",
        key="analysis_input_mode",
    )
    st.markdown('<hr class="editorial-rule">', unsafe_allow_html=True)

    if mode == "Manual Input":
        st.markdown('<div class="eyebrow">Input / spectral values</div>', unsafe_allow_html=True)
        st.subheader("Aggregate band powers")
        st.caption("Enter non-negative power values using the same scale across all five bands.")
        manual_bands = band_inputs()
        st.markdown(
            '<div class="empty-state">Manual values are projected equally across the model’s '
            '16 channels. This exploratory input does not reproduce channel-specific features '
            'from a raw recording.</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("Run Analysis", type="primary", key="analyze_manual"):
            if sum(manual_bands.values()) <= 0:
                st.error("Enter at least one band power greater than zero.")
            else:
                with st.spinner("Preparing features and running Experimental ML Analysis…"):
                    try:
                        model = load_model()
                        features = project_manual_bands(manual_bands)
                        result = predict(model, features)
                        save_analysis(
                            result,
                            manual_bands,
                            "Manual aggregate-band projection",
                            feature_values={key: float(value) for key, value in features.iloc[0].items()},
                        )
                        st.rerun()
                    except Exception as exc:
                        st.markdown(
                            '<div class="error-state"><strong>Analysis could not be completed.</strong><br>'
                            'Confirm that the model file is available and compatible with this application.</div>',
                            unsafe_allow_html=True,
                        )
                        with st.expander("Technical details"):
                            st.code(str(exc), language=None)

    else:
        st.markdown('<div class="eyebrow">Input / dataset</div>', unsafe_allow_html=True)
        st.subheader("Raw 16-channel EEG")
        st.caption("Expected sampling rate: 250 Hz · Minimum 250 rows · Extra columns are ignored")
        with st.expander("Required CSV input specifications", expanded=True):
            st.markdown(
                """
                **Required columns**

                `Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4`

                **Required value type**

                Numeric EEG samples only. Values must not contain blanks, text, NaN, or infinity.

                **Recording assumptions**

                250 Hz sampling rate · one row per sample · at least 250 rows · complete
                250-sample windows are analyzed as one-second epochs. Extra columns are allowed;
                an optional `status` column is ignored.
                """
            )
        uploaded = st.file_uploader("Drop a CSV here or browse", type=["csv"], key="eeg_upload")
        eeg = None
        if uploaded is None:
            st.markdown(
                '<div class="empty-state"><strong>No dataset selected.</strong><br>Upload a CSV '
                'with the 16 required EEG channels to validate and enable analysis.</div>',
                unsafe_allow_html=True,
            )
        if uploaded is not None:
            try:
                upload_id = (uploaded.file_id, uploaded.name, uploaded.size)
                cached_upload = st.session_state.get("validated_eeg_upload")
                if not cached_upload or cached_upload["id"] != upload_id:
                    dataframe = pd.read_csv(uploaded)
                    valid, errors, warnings, eeg = validate_raw_eeg(dataframe)
                    cached_upload = {
                        "id": upload_id,
                        "dataframe": dataframe,
                        "valid": valid,
                        "errors": errors,
                        "warnings": warnings,
                        "eeg": eeg,
                    }
                    st.session_state.validated_eeg_upload = cached_upload
                dataframe = cached_upload["dataframe"]
                valid = cached_upload["valid"]
                errors = cached_upload["errors"]
                warnings = cached_upload["warnings"]
                eeg = cached_upload["eeg"]
                if valid:
                    st.markdown(
                        f'<div class="empty-state"><strong>Feature validation complete.</strong><br>'
                        f'16 / 16 required channels detected · {len(eeg) // EPOCH:,} complete one-second record(s)</div>',
                        unsafe_allow_html=True,
                    )
                for warning in warnings:
                    st.warning(warning)
                for error in errors:
                    st.error(error)

                if valid and st.button(
                    "Analyze EEG and show final result",
                    type="primary",
                    key="analyze_csv",
                    width="stretch",
                ):
                    with st.spinner("Running the EEG model…"):
                        try:
                            model = load_model()
                            features = extract_features(eeg)
                            save_analysis(
                                predict(model, features),
                                estimate_bands_from_raw(eeg),
                                "Raw 16-channel CSV",
                                recording={
                                    "samples": len(eeg),
                                    "duration_seconds": len(eeg) / FS,
                                    "complete_epochs": len(features),
                                    "channels": len(CHANNELS),
                                },
                                feature_values={
                                    key: float(value) for key, value in features.mean().items()
                                },
                            )
                            st.rerun()
                        except Exception as exc:
                            st.markdown(
                                '<div class="error-state"><strong>Analysis could not be completed.</strong><br>'
                                'The uploaded data passed format checks, but model inference failed.</div>',
                                unsafe_allow_html=True,
                            )
                            with st.expander("Technical details"):
                                st.code(str(exc), language=None)

                st.dataframe(dataframe.head(10), width="stretch", hide_index=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("Rows", f"{len(dataframe):,}")
                c2.metric("Valid records", f"{len(dataframe) // EPOCH:,}" if valid else "0")
                c3.metric("Required channels", f"{sum(c in dataframe for c in CHANNELS)} / 16")
            except Exception as exc:
                st.markdown(
                    '<div class="error-state"><strong>The CSV could not be read.</strong><br>'
                    'Check that the file is a valid, comma-separated dataset with a header row.</div>',
                    unsafe_allow_html=True,
                )
                with st.expander("Technical details"):
                    st.code(str(exc), language=None)


def disease_page():
    page_intro(
        "Education / research context",
        "Alzheimer's Disease",
        "Understanding cognitive decline, neurological changes, and the role of computational research.",
    )
    st.markdown('<hr class="editorial-rule">', unsafe_allow_html=True)

    st.markdown('<div class="eyebrow">01 · Overview</div>', unsafe_allow_html=True)
    st.subheader("A progressive disease of the brain")
    st.markdown(
        '<div class="editorial-copy">Alzheimer’s disease is a progressive neurological '
        'condition that affects memory, thinking, and eventually the ability to carry out '
        'everyday activities. It is the most common cause of dementia—a general term for '
        'cognitive and functional decline severe enough to interfere with daily life. '
        'Because changes may develop over years and symptoms can have many causes, early '
        'research and careful assessment matter for understanding progression, separating '
        'possible explanations, and improving future detection and care.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="editorial-rule">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">02 · What changes in the brain?</div>', unsafe_allow_html=True)
    st.subheader("A complex, evolving biological process")
    st.markdown(
        '<div class="editorial-copy" style="margin-bottom:1.25rem">Alzheimer’s cannot be '
        'reduced to one molecule or one pathway. Researchers study interacting changes that '
        'develop over time, vary between people, and may begin before noticeable symptoms.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="science-grid">'
        '<div class="science-item"><strong>Amyloid-beta</strong><span>Fragments can accumulate '
        'between neurons as plaques. Their presence is an important feature of Alzheimer’s '
        'biology, but does not by itself explain every symptom or outcome.</span></div>'
        '<div class="science-item"><strong>Tau</strong><span>Altered tau can form tangles inside '
        'neurons and disrupt internal transport. Its distribution is studied in relation to '
        'neurodegeneration and cognitive change.</span></div>'
        '<div class="science-item"><strong>Neuronal dysfunction</strong><span>Neurons may lose '
        'their ability to communicate, maintain normal metabolism, and respond effectively '
        'within affected circuits.</span></div>'
        '<div class="science-item"><strong>Synaptic change</strong><span>Connections between '
        'neurons can weaken or disappear, affecting how information is transmitted and '
        'stored across brain regions.</span></div>'
        '<div class="science-item"><strong>Network change</strong><span>Coordinated activity '
        'across memory, attention, and other large-scale brain networks may become altered '
        'as disease processes evolve.</span></div>'
        '<div class="science-item"><strong>Neurodegeneration</strong><span>Progressive injury '
        'and loss of neurons can lead to tissue shrinkage and increasingly widespread '
        'functional impairment.</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="editorial-rule">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">03 · EEG & Alzheimer’s research</div>', unsafe_allow_html=True)
    st.subheader("Studying the rhythms of electrical brain activity")
    st.markdown(
        '<div class="editorial-copy">Electroencephalography (EEG) measures electrical '
        'activity recorded at the scalp. Researchers use spectral analysis to study how '
        'signal power is distributed across frequency bands in Alzheimer’s disease, mild '
        'cognitive impairment, healthy aging, and other neurological conditions.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown(
        '<table class="band-table"><thead><tr><th>Band</th><th>Approximate range</th>'
        '<th>Research description</th></tr></thead><tbody>'
        '<tr><td>Delta</td><td>0.5–4 Hz</td><td>Slow activity, prominent in deep sleep and '
        'studied in relation to diffuse slowing.</td></tr>'
        '<tr><td>Theta</td><td>4–8 Hz</td><td>Low-frequency activity associated with several '
        'cognitive and physiological states.</td></tr>'
        '<tr><td>Alpha</td><td>8–13 Hz</td><td>A prominent resting rhythm often strongest over '
        'posterior regions in relaxed wakefulness.</td></tr>'
        '<tr><td>Beta</td><td>13–30 Hz</td><td>Faster activity linked with active processing, '
        'attention, and sensorimotor function.</td></tr>'
        '<tr><td>Gamma</td><td>30–45+ Hz</td><td>Higher-frequency activity studied in local '
        'processing and coordinated neural activity.</td></tr>'
        '</tbody></table>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="notice" style="margin-top:1rem"><strong>Important context.</strong> '
        'Studies have investigated changes in spectral power, frequency ratios, connectivity, '
        'and shifts toward slower EEG activity. None of these patterns is individually '
        'diagnostic of Alzheimer’s disease. EEG can vary with age, alertness, sleep, medication, '
        'recording conditions, and many neurological or medical factors.</div>',
        unsafe_allow_html=True,
    )

    st.write("")
    relation_left, relation_right = st.columns([0.45, 1.55], gap="large", vertical_alignment="center")
    with relation_left:
        st.markdown(
            '<div class="eyebrow">Conceptual sequence</div><div class="editorial-copy">'
            'Biological, network, and electrophysiological observations operate at different '
            'levels of explanation.</div>',
            unsafe_allow_html=True,
        )
    with relation_right:
        st.markdown('<div class="eyebrow">Research relationship</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="vertical-flow">'
            '<div class="vertical-step"><span class="num">01</span><strong>Neurobiological change</strong></div>'
            '<div class="vertical-arrow">↓</div>'
            '<div class="vertical-step"><span class="num">02</span><strong>Synaptic & network disruption</strong></div>'
            '<div class="vertical-arrow">↓</div>'
            '<div class="vertical-step"><span class="num">03</span><strong>Altered population-level electrical activity</strong></div>'
            '<div class="vertical-arrow">↓</div>'
            '<div class="vertical-step"><span class="num">04</span><strong>EEG features studied computationally</strong></div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.caption("This is a research framework, not a deterministic causal chain or diagnostic pathway.")

    st.markdown('<hr class="editorial-rule">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">04 · EEG machine-learning research</div>', unsafe_allow_html=True)
    st.subheader("From supplied signal features to an experimental output")
    flow_col, copy_col = st.columns([0.8, 1.2], gap="large")
    with flow_col:
        st.markdown(
            '<div class="vertical-flow">'
            '<div class="vertical-step"><span class="num">01</span><strong>EEG Features</strong></div>'
            '<div class="vertical-arrow">↓</div>'
            '<div class="vertical-step"><span class="num">02</span><strong>Feature Validation</strong></div>'
            '<div class="vertical-arrow">↓</div>'
            '<div class="vertical-step"><span class="num">03</span><strong>Random Forest Model</strong></div>'
            '<div class="vertical-arrow">↓</div>'
            '<div class="vertical-step"><span class="num">04</span><strong>Experimental Pattern Similarity</strong></div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with copy_col:
        st.markdown(
            '<div class="editorial-copy">The prototype validates supplied EEG data, derives '
            'model-compatible features, and compares them with statistical patterns learned '
            'by a trained Random Forest classifier. The result is a <strong>research model '
            'output</strong>: a classification probability describing similarity to learned '
            'classes. It is not a <strong>clinical diagnosis</strong>, a measure of disease '
            'severity, or an estimate of an individual’s medical risk.</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        st.button("Open EEG Analysis  →", on_click=navigate, args=("EEG Analysis",))

    st.markdown('<hr class="editorial-rule">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">05 · Clinical Alzheimer\'s assessment</div>', unsafe_allow_html=True)
    st.subheader("A multi-source evaluation")
    st.markdown(
        '<div class="editorial-copy" style="margin-bottom:1rem">In real clinical practice, '
        'evaluation of cognitive concerns may draw on several sources of information. The '
        'appropriate combination depends on the person and the clinical setting.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="assessment-list">'
        '<div class="assessment-item">Clinical and symptom history</div>'
        '<div class="assessment-item">Cognitive and functional assessment</div>'
        '<div class="assessment-item">Neurological and physical evaluation</div>'
        '<div class="assessment-item">Laboratory testing</div>'
        '<div class="assessment-item">Structural or functional imaging</div>'
        '<div class="assessment-item">Biomarkers, where clinically appropriate</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.caption("This overview is educational and does not provide individualized diagnostic guidance.")

    st.markdown('<hr class="editorial-rule">', unsafe_allow_html=True)
    st.markdown(
        '<div class="peach-panel"><div class="eyebrow">06 · Research disclaimer</div>'
        '<strong>This application is an experimental research and educational prototype. '
        'Its machine-learning outputs have not been established here as a clinically '
        'validated diagnostic test and should not be interpreted as a diagnosis of '
        'Alzheimer’s disease.</strong></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="source-note" style="margin-top:1rem">Educational framing informed by '
        '<a href="https://www.nia.nih.gov/health/alzheimers-disease-fact-sheet" target="_blank">'
        'the National Institute on Aging</a> and a '
        '<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC6200063/" target="_blank">'
        'peer-reviewed systematic review of resting-state EEG research</a>.</div>',
        unsafe_allow_html=True,
    )


def about_page():
    page_intro(
        "Technical documentation",
        "About the Model",
        "A transparent guide to the machine-learning component, its inputs, execution path, and boundaries.",
    )
    st.write("")
    st.markdown('<span class="status-badge">Research / Educational Prototype</span>', unsafe_allow_html=True)

    try:
        model = load_model()
        classifier = getattr(model, "named_steps", {}).get("classifier")
    except Exception as exc:
        model, classifier = None, None
        st.error(f"Model metadata is unavailable because the artifact could not be loaded: {exc}")

    st.markdown('<hr class="editorial-rule">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Model overview</div>', unsafe_allow_html=True)
    st.subheader("Random Forest Classifier")
    st.markdown(
        '<div class="docs-block">A Random Forest combines many decision trees rather than '
        'relying on a single sequence of rules. Each tree evaluates the input features and '
        'contributes a class estimate; the ensemble combines those estimates into a more '
        'stable classification. This approach can be useful for structured scientific data '
        'because it can represent nonlinear relationships and interactions between features '
        'without requiring one tree to carry the full decision.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="editorial-rule">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Input features</div>', unsafe_allow_html=True)
    st.subheader(f"Configured feature vector · {len(FEATURES)} values")
    st.write(
        "This inventory is rendered directly from the same ordered `FEATURES` configuration "
        "used by the analysis service. It is not a separately maintained documentation list."
    )
    st.markdown(
        '<div class="feature-list">'
        + "".join(f'<div class="feature-chip">{feature}</div>' for feature in FEATURES)
        + '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="editorial-rule">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Pipeline</div>', unsafe_allow_html=True)
    st.subheader("Inference path")
    stages = [
        "EEG-derived values",
        "validation",
        "ordered feature vector",
        "Random Forest",
        "predict_proba()",
        "experimental classification output",
    ]
    pipeline_html = '<div class="pipeline-strip">'
    for index, stage in enumerate(stages):
        pipeline_html += f'<div class="pipeline-node">{stage}</div>'
        if index < len(stages) - 1:
            pipeline_html += '<div class="pipeline-link">→</div>'
    pipeline_html += '</div>'
    st.markdown(pipeline_html, unsafe_allow_html=True)
    st.caption(
        "Raw CSV recordings are validated and transformed into the configured feature order. "
        "The loaded pipeline then returns class probabilities; the interface presents them "
        "only as experimental model output."
    )

    st.markdown('<hr class="editorial-rule">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Model metadata</div>', unsafe_allow_html=True)
    st.subheader("Loaded artifact")
    metadata = []
    if classifier is not None:
        metadata.append(("Classifier type", type(classifier).__name__))
        if hasattr(classifier, "n_estimators"):
            metadata.append(("Number of estimators", classifier.n_estimators))
    if model is not None:
        if hasattr(model, "n_features_in_"):
            metadata.append(("Expected features", model.n_features_in_))
        if hasattr(model, "classes_"):
            metadata.append(("Class labels", ", ".join(str(value) for value in model.classes_)))

    if metadata:
        st.markdown(
            '<div class="metadata-grid">'
            + "".join(
                f'<div class="metadata-row"><div class="metadata-label">{label}</div>'
                f'<div class="metadata-value">{value}</div></div>'
                for label, value in metadata
            )
            + '</div>',
            unsafe_allow_html=True,
        )
    if model is not None and hasattr(model, "feature_names_in_"):
        with st.expander("feature_names_in_"):
            st.code("\n".join(str(value) for value in model.feature_names_in_), language=None)
    st.caption("Only metadata exposed by the currently loaded model artifact is shown.")

    st.markdown('<hr class="editorial-rule">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Limitations</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="limitations-panel"><strong>Interpret within the boundaries of the model.</strong>'
        '<ul><li>Predictions depend on the composition, quality, labels, and biases of the training data.</li>'
        '<li>EEG measurements can vary with acquisition hardware, montage, recording conditions, artifacts, and preprocessing.</li>'
        '<li>Distribution shift between training data and new inputs can materially affect model behavior.</li>'
        '<li>Machine-learning classification is not equivalent to a clinical diagnosis.</li>'
        '<li>Independent, appropriately designed validation is necessary before any clinical use.</li></ul></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="editorial-rule">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Research status</div>', unsafe_allow_html=True)
    st.subheader("Research / Educational Prototype")
    st.write(
        "This application supports exploration of an EEG machine-learning workflow. It "
        "does not claim clinical validation, provide a medical diagnosis, or replace "
        "assessment by qualified health professionals."
    )
    disclaimer()


def main():
    st.set_page_config(
        page_title="CognitiveNeuro | EEG Computational Analysis",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    initialize_state()
    inject_styles()
    product_navigation()
    pages = {
        "Overview": overview_page,
        "EEG Analysis": analysis_page,
        "Alzheimer's Disease": disease_page,
        "About the Model": about_page,
    }
    pages.get(st.session_state.page, overview_page)()
    render_analysis_assistant()


if __name__ == "__main__":
    main()
