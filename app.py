"""
Insect AI - Medical Vector & Pest Recognition System
Deep Learning Inference Web Application powered by Streamlit and TensorFlow/Keras.
"""

import os
import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd

from src.config import (
    CLASS_NAMES,
    SPECIES_METADATA,
    BENCHMARK_RESULTS,
    IMG_SIZE,
    DEFAULT_RELEASE_URL
)
from src.model_loader import load_insect_model
from src.utils import preprocess_image, predict_top_k

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Insect AI | Medical Vector & Pest Recognition",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling badges and cards
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #ff4b4b;
        margin-bottom: 12px;
    }
    .risk-severe { color: #dc3545; font-weight: bold; }
    .risk-high { color: #fd7e14; font-weight: bold; }
    .risk-moderate { color: #ffc107; font-weight: bold; }
    .risk-low { color: #28a745; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1fab3.png", width=64)
    st.title("Insect AI")
    st.caption("Deep Learning Vector Surveillance System")
    st.divider()

    st.subheader("⚙️ System Status")
    
    # Model Loading with Keras 2/3 multi-engine compatibility
    with st.spinner("Connecting to model weights..."):
        model, active_path, error = load_insect_model()

    # Dynamic inspection of model architecture
    eval_size = IMG_SIZE
    num_model_classes = len(CLASS_NAMES)
    if model is not None:
        try:
            inp = getattr(model, "input_shape", None)
            if inp and len(inp) == 4 and inp[1] is not None:
                eval_size = (int(inp[2]), int(inp[1]))
            out_shape = getattr(model, "output_shape", None)
            if out_shape and len(out_shape) >= 2:
                num_model_classes = int(out_shape[-1])
        except Exception:
            pass

    st.markdown(f"**Target Input Shape:** `{eval_size[0]} × {eval_size[1]} px`")
    st.markdown(f"**Model Output Classes:** `{num_model_classes} classes`")

    if error:
        st.error(f"❌ Failed to load model: {error}")
        st.info("You can upload a `.keras` file below or verify the download URL.")
    else:
        st.success(f"✅ Model active: `{os.path.basename(active_path)}`")
        if num_model_classes != len(CLASS_NAMES):
            st.warning(f"ℹ️ Model has {num_model_classes} outputs (insect dataset has {len(CLASS_NAMES)}). Running in adaptive mode.")

    st.divider()
    
    # Advanced Options
    with st.expander("🛠️ Advanced Settings", expanded=False):
        top_k = st.slider("Top Candidates (K)", min_value=1, max_value=8, value=5)
        rescale_pixel = st.checkbox("Rescale pixel values (/255.0)", value=True,
                                    help="Uncheck if your .keras model already includes an internal Rescaling layer.")
        
        st.markdown("---")
        st.markdown("**Custom Model Upload**")
        uploaded_model_file = st.file_uploader("Upload custom .keras model", type=["keras", "h5"])
        if uploaded_model_file:
            temp_model_path = f"custom_{uploaded_model_file.name}"
            with open(temp_model_path, "wb") as f:
                f.write(uploaded_model_file.getbuffer())
            st.success(f"Uploaded `{uploaded_model_file.name}`. Reloading...")
            st.cache_resource.clear()
            st.rerun()

    st.divider()
    st.markdown("### 🔗 Project Links")
    st.markdown("[📁 GitHub Repository](https://github.com/maduekedickson/insect-cnn-baseline)")
    st.markdown(f"[📦 Model Release Asset]({DEFAULT_RELEASE_URL})")
    st.caption("Designed for public health vector biosurveillance & pest control research.")


# ==========================================
# MAIN INTERFACE TABS
# ==========================================
tab_classifier, tab_benchmarks, tab_catalog, tab_docs = st.tabs([
    "🔍 Image Classifier",
    "📊 Research & Benchmarks",
    "🦟 Species Catalog (16)",
    "📖 Documentation & Setup"
])


# ------------------------------------------
# TAB 1: CLASSIFIER
# ------------------------------------------
with tab_classifier:
    st.title("🦟 Insect & Medical Vector Classifier")
    st.write(
        "Upload a photograph of a mosquito, tick, flea, or pest to determine its species, "
        "pathogen transmission potential, and public health risk level."
    )
    
    st.markdown("#### Step 1: Select or Upload an Insect Image")
    
    col_upload, col_sample = st.columns([2, 1])
    
    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload image file",
            type=["jpg", "jpeg", "png"],
            help="High-contrast, focused macro photos yield the highest accuracy."
        )

    with col_sample:
        st.markdown("**Or test with sample images:**")
        sample_options = {
            "None": None,
            "Aedes aegypti (Yellow Fever Mosquito)": "sample_images/sample_aedes_aegypti.jpg",
            "Anopheles stephensi (Malaria Mosquito)": "sample_images/sample_anopheles_stephensi.jpg",
            "Ixodes scapularis (Deer Tick)": "sample_images/sample_ixodes_scapularis.jpg"
        }
        selected_sample = st.selectbox("Choose sample", list(sample_options.keys()))

    active_image = None
    image_source_name = ""

    if uploaded_file is not None:
        active_image = Image.open(uploaded_file)
        image_source_name = uploaded_file.name
    elif selected_sample != "None" and sample_options[selected_sample] is not None:
        sample_path = sample_options[selected_sample]
        if os.path.exists(sample_path):
            active_image = Image.open(sample_path)
            image_source_name = os.path.basename(sample_path)

    st.divider()

    if active_image is not None:
        col_img, col_pred = st.columns([1, 1], gap="large")

        with col_img:
            st.subheader("📸 Input Image")
            st.image(active_image, caption=f"Source: {image_source_name}", use_container_width=True)
            w, h = active_image.size
            st.caption(f"Original Dimensions: {w} × {h} px | Evaluated at: {eval_size[0]} × {eval_size[1]} px")

        with col_pred:
            st.subheader("🎯 Identification Results")

            if model is None:
                st.error("Model weights are not loaded. Please verify the model download in the sidebar.")
            else:
                with st.spinner("Analyzing morphology and extracting visual features..."):
                    processed_array = preprocess_image(
                        active_image,
                        target_size=eval_size,
                        rescale=rescale_pixel
                    )
                    predictions = predict_top_k(model, processed_array, class_names=CLASS_NAMES, k=top_k)

                top_pred = predictions[0]
                meta = top_pred["metadata"]
                risk = meta.get("risk_level", "Unknown")
                
                # Risk color mapping
                risk_color_map = {
                    "Critical": "risk-severe",
                    "Severe": "risk-severe",
                    "High": "risk-high",
                    "Moderate": "risk-moderate",
                    "Low": "risk-low"
                }
                risk_class = risk_color_map.get(risk, "risk-moderate")

                # Top Result Highlight Banner
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin: 0; color: #1f2937;">{meta.get('common_name', top_pred['class_name'])}</h3>
                    <p style="font-style: italic; color: #4b5563; margin-top: 4px;">Scientific / Class: <b>{top_pred['class_name'].replace('_', ' ')}</b></p>
                    <p>Confidence: <b style="font-size: 1.2em; color: #ff4b4b;">{top_pred['confidence_percent']}</b></p>
                    <p>Vector Status: <b>{meta.get('vector_status', 'N/A')}</b> | Risk: <span class="{risk_class}">{risk}</span></p>
                </div>
                """, unsafe_allow_html=True)

                st.progress(top_pred["probability"])

                # Top-K Probability Breakdown Chart
                st.markdown("##### Probability Distribution")
                chart_data = pd.DataFrame({
                    "Species": [p["metadata"].get("common_name", p["class_name"]) for p in predictions],
                    "Confidence (%)": [round(p["probability"] * 100, 2) for p in predictions]
                }).set_index("Species")
                st.bar_chart(chart_data)

                # Medical Entomology Dossier
                with st.expander("📋 Medical Entomology & Epidemiological Profile", expanded=True):
                    st.markdown(f"**Taxonomic Group:** {meta.get('category', 'N/A')}")
                    st.markdown(f"**Ecological Habitat:** {meta.get('habitat', 'N/A')}")
                    
                    pathogens = meta.get("pathogens", [])
                    if pathogens:
                        st.markdown("**Pathogens / Clinical Associations:**")
                        for p in pathogens:
                            st.markdown(f"- 🦠 *{p}*")
                    else:
                        st.markdown("No significant viral or parasitic pathogens reported.")
    else:
        st.info("👆 Please upload an image or select one of the sample presets above to view predictions.")


# ------------------------------------------
# TAB 2: RESEARCH & BENCHMARKS
# ------------------------------------------
with tab_benchmarks:
    st.header("📊 Model Evaluation & Experimental Benchmarks")
    st.write(
        "Performance metrics from the comparative study evaluating a custom baseline CNN "
        "trained from scratch against pretrained architectures (MobileNetV2, EfficientNetB0)."
    )

    st.subheader("1. Comprehensive Model Comparison")
    df_models = pd.DataFrame(BENCHMARK_RESULTS["Overall_Models"])
    st.dataframe(df_models.style.format({
        "Accuracy": "{:.2%}",
        "Precision": "{:.2%}",
        "Recall": "{:.2%}",
        "Macro_F1": "{:.2%}"
    }), use_container_width=True)

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Baseline CNN Accuracy", "85.42%", "Trained from scratch")
    col_m2.metric("MobileNetV2 Accuracy", "90.03%", "+4.61% gain (Best overall)")
    col_m3.metric("EfficientNetB0 Accuracy", "89.61%", "+4.19% gain")

    st.divider()

    st.subheader("2. Dataset Scaling Sensitivity Analysis")
    st.write(
        "Evaluating model resilience when training data is restricted to subsets (25%, 50%, and 100%). "
        "Transfer learning models maintained high accuracy even under extreme data scarcity, "
        "whereas the baseline CNN required complete data representation to converge."
    )

    df_scaling = pd.DataFrame(BENCHMARK_RESULTS["Dataset_Scaling"]).set_index("Dataset_Size")
    st.line_chart(df_scaling)
    
    # Modern pandas compatible table formatting (map vs applymap)
    formatter = lambda v: f"{v * 100:.2f}%" if isinstance(v, (int, float)) else str(v)
    formatted_scaling = df_scaling.map(formatter) if hasattr(df_scaling, "map") else df_scaling.applymap(formatter)
    st.table(formatted_scaling)


# ------------------------------------------
# TAB 3: SPECIES CATALOG
# ------------------------------------------
with tab_catalog:
    st.header("🦟 Monitored Insect Species Directory")
    st.write("Reference database of the 16 insect species and disease vectors recognized by this neural network.")

    filter_risk = st.selectbox("Filter by Public Health Risk Level", ["All", "Critical", "Severe", "High", "Moderate", "Low"])

    for name, data in SPECIES_METADATA.items():
        if filter_risk != "All" and data.get("risk_level") != filter_risk:
            continue

        with st.container():
            st.markdown(f"### {data.get('common_name')} (*{name.replace('_', ' ')}*)")
            col1, col2, col3 = st.columns([1, 1, 2])
            col1.markdown(f"**Category:** {data.get('category')}")
            col2.markdown(f"**Risk Level:** `{data.get('risk_level')}`")
            col3.markdown(f"**Vector Status:** {data.get('vector_status')}")
            
            st.markdown(f"**Habitat:** {data.get('habitat')}")
            pathogens = data.get("pathogens", [])
            st.markdown(f"**Known Pathogens / Health Concerns:** {', '.join(pathogens)}")
            st.divider()


# ------------------------------------------
# TAB 4: DOCUMENTATION & SETUP
# ------------------------------------------
with tab_docs:
    st.header("📖 Documentation & Reproducibility Guide")
    
    st.markdown("""
    ### 🏗️ Convolutional Architecture
    The baseline model is a 4-block Convolutional Neural Network trained from scratch:
    - **Input Layer:** `(224, 224, 3)`
    - **Data Augmentation:** Random Flip, Rotation, and Zoom layers
    - **Block 1:** Conv2D (32 filters, 3x3) + BatchNorm + MaxPooling2D
    - **Block 2:** Conv2D (64 filters, 3x3) + BatchNorm + MaxPooling2D
    - **Block 3:** Conv2D (128 filters, 3x3) + BatchNorm + MaxPooling2D
    - **Block 4:** Conv2D (256 filters, 3x3) + BatchNorm + MaxPooling2D
    - **Classifier:** GlobalAveragePooling2D + Dense(256, ReLU) + Dropout(0.5) + Dense(16, Softmax)

    ### 🚀 Local Quickstart
    ```powershell
    # Clone repository
    git clone https://github.com/maduekedickson/insect-cnn-baseline.git
    cd insect-cnn-baseline

    # Create & activate environment
    py -3.12 -m venv .venv
    .\\.venv\\Scripts\\Activate.ps1

    # Install dependencies
    pip install -r requirements.txt

    # Run application
    streamlit run app.py
    ```

    ### ☁️ Streamlit Cloud Deployment
    This app is engineered to deploy seamlessly on **Streamlit Community Cloud**:
    1. Repository: `maduekedickson/insect-cnn-baseline`
    2. Main file path: `app.py`
    3. Model weights: Downloaded automatically from GitHub Releases CDN (`trained_model.keras`).
    """)
