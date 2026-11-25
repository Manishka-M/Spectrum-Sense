
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ==================================================
# 1. PROJECT PATHS AND IMPORTS
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.signal_generator import (
    generate_bpsk,
    generate_qpsk,
    generate_8psk,
    generate_16qam,
    add_awgn,
)

from src.feature_extractor import extract_features
from src.predict import load_model


MODEL_PATH = ROOT / "models" / "best_model.joblib"


# ==================================================
# 2. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="SpectrumSense | RF Signal Classifier",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# 3. CUSTOM DESIGN AND STYLING
# ==================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background:
            radial-gradient(
                ellipse at 50% -10%,
                rgba(14, 165, 233, 0.16),
                transparent 48%
            ),
            linear-gradient(
                180deg,
                #0b1120 0%,
                #0b1120 60%,
                #0e1728 100%
            );
        color: #e5e7eb;
    }

    /* Main content width */
    .block-container {
        max-width: 1250px;
        padding-top: 6.2rem;
        padding-bottom: 2rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #111c30,
            #0d1728
        );
        border-right: 1px solid rgba(148, 163, 184, 0.12);
    }

    /* Main title */
    h1 {
        color: #f8fafc !important;
        font-size: 3.4rem !important;
        font-weight: 800 !important;
        letter-spacing: -1.8px !important;
        line-height: 1.15 !important;
        margin-bottom: 0.15rem !important;
    }

    /* Section headings */
    h2 {
        color: #f1f5f9 !important;
        font-size: 1.55rem !important;
        font-weight: 700 !important;
    }

    h3 {
        color: #e2e8f0 !important;
        font-size: 1.2rem !important;
        font-weight: 650 !important;
    }

    /* Header eyebrow */
    .eyebrow {
        color: #38bdf8;
        font-size: 0.82rem;
        font-weight: 750;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 14px;
    }

    /* Subtitle */
    .subtitle {
        color: #a5b4c8;
        font-size: 1.12rem;
        margin-top: 0.35rem;
    }

    /* Description */
    .description {
        color: #718198;
        font-size: 0.9rem;
        line-height: 1.7;
        margin-top: 0.5rem;
    }

    /* Header separator */
    .header-line {
        height: 1px;
        background: linear-gradient(
            90deg,
            rgba(56, 189, 248, 0.65),
            rgba(56, 189, 248, 0.12),
            transparent
        );
        margin-top: 25px;
        margin-bottom: 25px;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(
            145deg,
            #17243a,
            #111c2e
        );
        border: 1px solid rgba(148, 163, 184, 0.13);
        border-radius: 13px;
        padding: 16px 18px;
        box-shadow: 0 5px 18px rgba(0, 0, 0, 0.12);
    }

    div[data-testid="stMetricLabel"] {
        color: #94a3b8;
        font-size: 0.82rem;
    }

    div[data-testid="stMetricValue"] {
        color: #38bdf8;
        font-size: 1.8rem;
        font-weight: 750;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        color: #94a3b8;
        font-weight: 600;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #38bdf8;
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 10px;
        overflow: hidden;
    }

    /* Dividers */
    hr {
        border-color: rgba(148, 163, 184, 0.15) !important;
        margin-top: 24px;
        margin-bottom: 24px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.8rem;
        line-height: 1.8;
        padding: 25px 0 10px 0;
        border-top: 1px solid rgba(148, 163, 184, 0.12);
    }

    /* Mobile */
    @media (max-width: 768px) {
        h1 {
            font-size: 2.5rem !important;
        }

        .block-container {
            padding-top: 1.3rem;
        }

        .eyebrow {
            font-size: 0.7rem;
            letter-spacing: 2px;
        }

        .subtitle {
            font-size: 0.95rem;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# 4. MODULATION THEORY
# ==================================================

THEORY = {
    "BPSK": {
        "name": "Binary Phase Shift Keying",
        "description": (
            "BPSK is a digital modulation technique in which two distinct "
            "phase states represent binary information. The two phases "
            "are separated by 180 degrees."
        ),
        "equation": r"s(t) = A\cos(2\pi f_c t + \phi_i)",
        "bits": "1 bit per symbol",
        "symbols": "2",
        "phase": "0° and 180°",
        "amplitude": "Constant",
        "advantages": "High noise immunity and simple implementation",
        "limitations": "Lower spectral efficiency",
        "applications": "Satellite communication and robust digital links",
        "points": [
            (1, 0),
            (-1, 0),
        ],
    },

    "QPSK": {
        "name": "Quadrature Phase Shift Keying",
        "description": (
            "QPSK uses four distinct phase states to represent digital "
            "information. Each symbol carries two bits, improving "
            "spectral efficiency compared with BPSK."
        ),
        "equation": r"s(t) = A\cos(2\pi f_c t + \phi_i)",
        "bits": "2 bits per symbol",
        "symbols": "4",
        "phase": "45°, 135°, 225°, 315°",
        "amplitude": "Constant",
        "advantages": "Better spectral efficiency than BPSK",
        "limitations": "More sensitive to phase errors than BPSK",
        "applications": "Satellite, Wi-Fi and cellular communication",
        "points": [
            (np.cos(np.pi / 4), np.sin(np.pi / 4)),
            (np.cos(3 * np.pi / 4), np.sin(3 * np.pi / 4)),
            (np.cos(5 * np.pi / 4), np.sin(5 * np.pi / 4)),
            (np.cos(7 * np.pi / 4), np.sin(7 * np.pi / 4)),
        ],
    },

    "8PSK": {
        "name": "8-Phase Shift Keying",
        "description": (
            "8PSK uses eight distinct phase states to represent digital "
            "information. Each symbol carries three bits, increasing "
            "spectral efficiency compared with BPSK and QPSK."
        ),
        "equation": r"s(t) = A\cos(2\pi f_c t + \phi_i)",
        "bits": "3 bits per symbol",
        "symbols": "8",
        "phase": "Eight equally spaced phase states",
        "amplitude": "Constant",
        "advantages": "Higher data rate than BPSK and QPSK",
        "limitations": "More sensitive to noise and phase errors",
        "applications": "Satellite and bandwidth-efficient communication",
        "points": [
            (
                np.cos(2 * np.pi * i / 8),
                np.sin(2 * np.pi * i / 8),
            )
            for i in range(8)
        ],
    },

    "16QAM": {
        "name": "16-Quadrature Amplitude Modulation",
        "description": (
            "16QAM uses sixteen different combinations of amplitude "
            "and phase. Each symbol represents four bits. Unlike PSK, "
            "both amplitude and phase vary."
        ),
        "equation": r"s(t) = I(t) + jQ(t)",
        "bits": "4 bits per symbol",
        "symbols": "16",
        "phase": "Multiple phase values",
        "amplitude": "Variable",
        "advantages": "High spectral efficiency",
        "limitations": "More sensitive to noise and amplitude distortion",
        "applications": "Wi-Fi, LTE and broadband communication",
        "points": [
            (i / np.sqrt(10), q / np.sqrt(10))
            for i in [-3, -1, 1, 3]
            for q in [-3, -1, 1, 3]
        ],
    },
}


# ==================================================
# 5. MODEL LOADING
# ==================================================

@st.cache_resource
def get_model():
    if not MODEL_PATH.exists():
        return None

    return load_model(str(MODEL_PATH))


artifact = get_model()


# ==================================================
# 6. HEADER - NATIVE STREAMLIT COMPONENTS
# ==================================================

st.markdown(
    '<div class="eyebrow">INTELLIGENT RF ANALYSIS SYSTEM</div>',
    unsafe_allow_html=True,
)

# Native Streamlit title. No HTML title markup.
st.title("SpectrumSense")

st.markdown(
    '<div class="subtitle">'
    'AI-Powered RF Signal Classification System'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="description">'
    'Intelligent modulation recognition through signal processing, '
    'statistical feature extraction and machine learning.'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="header-line"></div>', unsafe_allow_html=True)


# ==================================================
# 7. SIDEBAR CONFIGURATION
# ==================================================

st.sidebar.title("⚙️ Signal Configuration")

st.sidebar.caption("Configure the input signal.")

modulation = st.sidebar.selectbox(
    "Select Modulation Scheme",
    ["BPSK", "QPSK", "8PSK", "16QAM"],
)

snr_db = st.sidebar.slider(
    "Signal-to-Noise Ratio (dB)",
    min_value=-5,
    max_value=20,
    value=10,
    step=1,
)

n_symbols = st.sidebar.slider(
    "Number of Symbols",
    min_value=128,
    max_value=1024,
    value=256,
    step=128,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Information")
st.sidebar.write("**Algorithm:** RBF Support Vector Machine")
st.sidebar.write("**Modulation Types:** 4")
st.sidebar.write("**Selected Features:** 9")
st.sidebar.write("**Test Accuracy:** 87.00%")


# ==================================================
# 8. PERFORMANCE METRICS
# ==================================================

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    st.metric("Test Accuracy", "87.00%")

with metric2:
    st.metric("Modulation Types", "4")

with metric3:
    st.metric("Selected Features", "9")

with metric4:
    st.metric("Classifier", "RBF-SVM")


# ==================================================
# 9. DYNAMIC TECHNICAL THEORY
# ==================================================

st.markdown("---")
st.header("Technical Theory")

selected_theory = THEORY[modulation]

st.subheader(f"{modulation} | {selected_theory['name']}")

st.write(selected_theory["description"])

st.markdown("**Mathematical Representation**")
st.latex(selected_theory["equation"])

theory_col1, theory_col2 = st.columns(2)

with theory_col1:
    st.markdown("#### Technical Characteristics")
    st.write(f"**Bits per symbol:** {selected_theory['bits']}")
    st.write(f"**Number of symbols:** {selected_theory['symbols']}")
    st.write(f"**Phase:** {selected_theory['phase']}")
    st.write(f"**Amplitude:** {selected_theory['amplitude']}")

with theory_col2:
    st.markdown("#### Performance and Applications")
    st.write(f"**Advantages:** {selected_theory['advantages']}")
    st.write(f"**Limitations:** {selected_theory['limitations']}")
    st.write(f"**Applications:** {selected_theory['applications']}")


# ==================================================
# 10. IDEAL CONSTELLATION
# ==================================================

st.markdown("#### Ideal Constellation Diagram")

ideal_points = np.array(selected_theory["points"])

fig_ideal, ax_ideal = plt.subplots(figsize=(4.5, 3.2))

ax_ideal.scatter(
    ideal_points[:, 0],
    ideal_points[:, 1],
    s=65,
    edgecolors="white",
)

ax_ideal.axhline(0, color="gray", linewidth=0.8)
ax_ideal.axvline(0, color="gray", linewidth=0.8)

ax_ideal.set_xlabel("In-phase (I)")
ax_ideal.set_ylabel("Quadrature (Q)")
ax_ideal.set_title(f"Ideal {modulation} Constellation")
ax_ideal.grid(True, alpha=0.25)
ax_ideal.set_aspect("equal", adjustable="box")

st.pyplot(fig_ideal, use_container_width=False)
plt.close(fig_ideal)


# ==================================================
# 11. SIGNAL GENERATION
# ==================================================

st.markdown("---")
st.header("Signal Classification")

st.write(
    "A synthetic complex-baseband signal is generated using the "
    "selected modulation scheme. AWGN is added at the selected SNR. "
    "The trained machine learning model then predicts its modulation."
)

generators = {
    "BPSK": generate_bpsk,
    "QPSK": generate_qpsk,
    "8PSK": generate_8psk,
    "16QAM": generate_16qam,
}

rng = np.random.default_rng()

# Keep the displayed signal constellation aligned with the ideal one.
phase_offset = 0.0

clean_signal = generators[modulation](
    n_symbols=n_symbols,
    rng=rng,
    phase_offset=phase_offset,
)

noisy_signal = add_awgn(
    clean_signal,
    snr_db=snr_db,
    rng=rng,
)


# ==================================================
# 12. FEATURE EXTRACTION AND PREDICTION
# ==================================================

if artifact is None:

    st.error(
        "Model file not found. Train the model first using "
        "`python -m src.train`."
    )

else:

    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    features = extract_features(noisy_signal)

    input_data = pd.DataFrame(
        [features],
        columns=features.keys(),
    )

    input_data = input_data[feature_columns]

    prediction = model.predict(input_data)[0]


    # ==================================================
    # 13. CLASSIFICATION RESULT
    # ==================================================

    st.subheader("Classification Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric("Selected Modulation", modulation)

    with result_col2:
        st.metric("Predicted Modulation", prediction)

    if prediction == modulation:
        st.success(
            f"Correct Classification: The model identified "
            f"the signal as {prediction}."
        )
    else:
        st.warning(
            f"Classification Mismatch: Expected {modulation}, "
            f"but the model predicted {prediction}."
        )


    # ==================================================
    # 14. SIGNAL ANALYSIS
    # ==================================================

    st.markdown("---")
    st.header("Signal Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Received Constellation",
            "Ideal vs Received",
            "Signal Statistics",
            "Extracted Features",
        ]
    )


    # --------------------------------------------------
    # TAB 1: RECEIVED CONSTELLATION
    # --------------------------------------------------

    with tab1:

        st.markdown("#### Received Noisy Signal")

        fig_received, ax_received = plt.subplots(
            figsize=(4.5, 3.2)
        )

        ax_received.scatter(
            np.real(noisy_signal),
            np.imag(noisy_signal),
            s=12,
            alpha=0.65,
        )

        ax_received.axhline(0, color="gray", linewidth=0.8)
        ax_received.axvline(0, color="gray", linewidth=0.8)

        ax_received.set_xlabel("In-phase (I)")
        ax_received.set_ylabel("Quadrature (Q)")
        ax_received.set_title(
            f"Received {modulation} at {snr_db} dB SNR"
        )

        ax_received.grid(True, alpha=0.25)
        ax_received.set_aspect("equal", adjustable="datalim")

        left, center, right = st.columns([1, 2, 1])

        with center:
            st.pyplot(
                fig_received,
                use_container_width=False,
            )

        plt.close(fig_received)


    # --------------------------------------------------
    # TAB 2: IDEAL VS RECEIVED
    # --------------------------------------------------

    with tab2:

        st.markdown("#### Ideal and Received Constellations")

        fig_compare, axes = plt.subplots(
            1,
            2,
            figsize=(8, 3.2),
        )

        axes[0].scatter(
            ideal_points[:, 0],
            ideal_points[:, 1],
            s=55,
            edgecolors="white",
        )

        axes[0].set_title(f"Ideal {modulation}")
        axes[0].set_xlabel("In-phase (I)")
        axes[0].set_ylabel("Quadrature (Q)")
        axes[0].grid(True, alpha=0.25)
        axes[0].axhline(0, color="gray", linewidth=0.8)
        axes[0].axvline(0, color="gray", linewidth=0.8)
        axes[0].set_aspect("equal", adjustable="datalim")

        axes[1].scatter(
            np.real(noisy_signal),
            np.imag(noisy_signal),
            s=12,
            alpha=0.65,
        )

        axes[1].set_title("Received Signal")
        axes[1].set_xlabel("In-phase (I)")
        axes[1].set_ylabel("Quadrature (Q)")
        axes[1].grid(True, alpha=0.25)
        axes[1].axhline(0, color="gray", linewidth=0.8)
        axes[1].axvline(0, color="gray", linewidth=0.8)
        axes[1].set_aspect("equal", adjustable="datalim")

        fig_compare.tight_layout()

        st.pyplot(
            fig_compare,
            use_container_width=False,
        )

        plt.close(fig_compare)


    # --------------------------------------------------
    # TAB 3: SIGNAL STATISTICS
    # --------------------------------------------------

    with tab3:

        st.markdown("#### Signal Statistics")

        avg_power = np.mean(np.abs(noisy_signal) ** 2)
        peak_power = np.max(np.abs(noisy_signal) ** 2)
        avg_amplitude = np.mean(np.abs(noisy_signal))
        peak_amplitude = np.max(np.abs(noisy_signal))

        stat1, stat2 = st.columns(2)

        with stat1:
            st.metric("Average Power", f"{avg_power:.4f}")
            st.metric("Average Amplitude", f"{avg_amplitude:.4f}")

        with stat2:
            st.metric("Peak Power", f"{peak_power:.4f}")
            st.metric("Peak Amplitude", f"{peak_amplitude:.4f}")

        st.write(f"**Number of Symbols:** {n_symbols}")
        st.write(f"**Selected SNR:** {snr_db} dB")
        st.write("**Signal Type:** Complex Baseband")


    # --------------------------------------------------
    # TAB 4: EXTRACTED FEATURES
    # --------------------------------------------------

    with tab4:

        st.markdown("#### Extracted Signal Features")

        feature_table = pd.DataFrame(
            {
                "Feature": list(features.keys()),
                "Value": [
                    round(float(value), 6)
                    for value in features.values()
                ],
            }
        )

        st.dataframe(
            feature_table,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "The trained model uses the nine selected features "
            "stored in its model artifact."
        )


# ==================================================
# 15. FOOTER
# ==================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">
        SpectrumSense | AI-Powered RF Signal Classification<br>
        Synthetic Complex-Baseband Simulation | RBF-SVM Classifier
    </div>
    """,
    unsafe_allow_html=True,
)
