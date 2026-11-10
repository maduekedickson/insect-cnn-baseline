"""
Model loading and downloading utilities with progress tracking.
Supports direct GitHub Release CDN download, local caching, and custom uploads.
"""

import os
import urllib.request
import streamlit as st
import keras
from src.config import DEFAULT_RELEASE_URL, LOCAL_MODEL_PATH

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
                # Update every 2% to avoid excessive rerenders
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


@st.cache_resource(show_spinner=False)
def load_insect_model(custom_model_path: str = None):
    """
    Loads and caches the Keras CNN model.
    Checks local directory first, then downloads from GitHub Releases.
    """
    model_path = custom_model_path or MODEL_FILENAME

    # Check local workspace root
    if not os.path.exists(model_path):
        # Also check fallback filename 'cnn_best.keras' if present
        if os.path.exists("cnn_best.keras"):
            model_path = "cnn_best.keras"
        else:
            with st.spinner("Downloading trained CNN weights from GitHub Releases..."):
                download_with_progress(DEFAULT_RELEASE_URL, model_path)

    try:
        model = keras.models.load_model(model_path, compile=False)
        return model, model_path, None
    except Exception as e:
        return None, model_path, str(e)
