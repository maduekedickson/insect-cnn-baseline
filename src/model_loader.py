"""
Model loading and downloading utilities with progress tracking.
Supports direct GitHub Release CDN download, local caching, Keras 2/3 compatibility, and custom uploads.
"""

import os
import urllib.request
import zipfile
import json
import streamlit as st
from src.config import DEFAULT_RELEASE_URL

MODEL_FILENAME = "trained_model.keras"


def download_with_progress(url: str, dest_path: str):
    """
    Downloads a model file from a URL with a Streamlit progress bar.
    """
    class DownloadProgressBar:
        def __init__(self, pbar, status_text):
            self.pbar = pbar
            self.status_text = status_text
            self.last_percent = 0

        def __call__(self, block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(1.0, downloaded / total_size)
                if percent - self.last_percent >= 0.02 or percent >= 1.0:
                    self.last_percent = percent
                    self.pbar.progress(percent)
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    self.status_text.caption(
                        f"📥 Downloading weights: {mb_downloaded:.1f} MB / {mb_total:.1f} MB ({int(percent * 100)}%)"
                    )

    progress_bar = st.progress(0.0)
    status_text = st.empty()
    try:
        opener = urllib.request.build_opener()
        opener.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) InsectClassifier/1.0")]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(
            url,
            dest_path,
            DownloadProgressBar(progress_bar, status_text)
        )
        status_text.caption("✅ Download complete! Initializing neural network...")
    finally:
        progress_bar.empty()
        status_text.empty()


def sanitize_keras2_config(model_path: str) -> str:
    """
    Removes legacy Keras 2 'batch_input_shape' from Conv2D and other layers in config.json
    so that Keras 3 can deserialize the model without throwing an Unrecognized keyword argument error.
    """
    if not model_path.endswith(".keras") or not os.path.exists(model_path):
        return model_path

    patched_path = model_path.replace(".keras", "_k3_compat.keras")
    if os.path.exists(patched_path):
        return patched_path

    try:
        with zipfile.ZipFile(model_path, "r") as zin:
            if "config.json" not in zin.namelist():
                return model_path
            cfg_bytes = zin.read("config.json")
            cfg = json.loads(cfg_bytes.decode("utf-8"))

            modified = False
            layers = cfg.get("config", {}).get("layers", [])
            for layer in layers:
                if layer.get("class_name") != "InputLayer":
                    l_cfg = layer.get("config", {})
                    if "batch_input_shape" in l_cfg:
                        del l_cfg["batch_input_shape"]
                        modified = True

            if not modified:
                return model_path

            with zipfile.ZipFile(patched_path, "w") as zout:
                for item in zin.infolist():
                    if item.filename == "config.json":
                        zout.writestr("config.json", json.dumps(cfg))
                    else:
                        zout.writestr(item, zin.read(item.filename))

        return patched_path
    except Exception:
        return model_path


@st.cache_resource(show_spinner=False)
def load_insect_model(custom_model_path: str = None):
    """
    Loads and caches the model with multi-engine compatibility (tf_keras, Keras 3, and auto-sanitization).
    """
    model_path = custom_model_path or MODEL_FILENAME

    if not os.path.exists(model_path):
        if os.path.exists("cnn_best.keras"):
            model_path = "cnn_best.keras"
        else:
            with st.spinner("Downloading trained weights from GitHub Releases..."):
                download_with_progress(DEFAULT_RELEASE_URL, model_path)

    # Strategy 1: Try legacy tf_keras (handles Keras 2.15 models natively)
    try:
        import tf_keras as legacy_keras
        model = legacy_keras.models.load_model(model_path, compile=False)
        return model, model_path, None
    except Exception as e_legacy:
        pass

    # Strategy 2: Try standard keras directly
    try:
        import keras
        model = keras.models.load_model(model_path, compile=False)
        return model, model_path, None
    except Exception as e_keras:
        pass

    # Strategy 3: Auto-sanitize Keras 2 config to Keras 3 format and reload
    try:
        import keras
        patched_path = sanitize_keras2_config(model_path)
        model = keras.models.load_model(patched_path, compile=False)
        return model, patched_path, None
    except Exception as e_final:
        return None, model_path, str(e_final)
