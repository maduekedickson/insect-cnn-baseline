# 🦟 Insect AI: Deep Learning Medical Vector & Pest Recognition System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21.0-orange.svg)](https://tensorflow.org/)
[![Model Release](https://img.shields.io/badge/Model-GitHub%20Release%20CDN-brightgreen)](https://github.com/maduekedickson/insect-cnn-baseline/releases/tag/deep_learning)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end deep learning system and web application designed for automated recognition of **16 insect species and disease vectors** of critical public health and agricultural concern. 

---

## 📌 Public Health Context & Problem Statement

Vector-borne diseases account for more than **17% of all infectious diseases globally**, causing over **700,000 deaths each year**. Pathogens transmitted by hematophagous arthropods (mosquitoes, ticks, fleas, and lice) include:
* **Malaria** (*Plasmodium* spp. via *Anopheles* mosquitoes)
* **Dengue, Zika, and Yellow Fever** (Flaviviruses via *Aedes* mosquitoes)
* **Lymphatic Filariasis & West Nile Virus** (via *Culex* mosquitoes)
* **Lyme Borreliosis & Tick-Borne Encephalitis** (via *Ixodes* ticks)
* **Epidemic Typhus** (*Rickettsia prowazekii* via *Pediculus humanus corporis*)

Traditional morphological identification requires specialized entomological training and microscopic inspection. **Insect AI** bridges this gap by deploying computer vision models to provide instantaneous, accessible species classification and epidemiological risk assessment from macro imagery.

---

## 🔬 Monitored Species & Medical Taxonomy

The system classifies images into 16 distinct arthropod taxa:

| Category | Scientific Name | Common Name | Primary Pathogen / Medical Concern | Risk Level |
| :--- | :--- | :--- | :--- | :--- |
| **Mosquito** | *Aedes aegypti* | Yellow Fever Mosquito | Dengue, Zika, Chikungunya, Yellow Fever | 🔴 Severe |
| **Mosquito** | *Aedes albopictus* | Asian Tiger Mosquito | Dengue, Chikungunya, Zika Virus | 🔴 Severe |
| **Mosquito** | *Aedes vexans* | Inland Floodwater Mosquito | Rift Valley Fever, West Nile Virus | 🟡 Moderate |
| **Mosquito** | *Anopheles stephensi* | Indo-Pakistan Malaria Mosquito | *Plasmodium falciparum*, *Plasmodium vivax* | 🔴 Critical |
| **Mosquito** | *Anopheles tessellatus* | Tessellated Mosquito | Secondary Malaria Vector | 🟡 Moderate |
| **Mosquito** | *Culex quinquefasciatus* | Southern House Mosquito | Lymphatic Filariasis, West Nile Virus | 🔴 Severe |
| **Mosquito** | *Culex vishnui* | Vishnui Mosquito | Japanese Encephalitis Virus (JEV) | 🔴 Severe |
| **Tick** | *Amblyomma americanum* | Lone Star Tick | Ehrlichiosis, Heartland Virus, Alpha-Gal Allergy | 🟠 High |
| **Tick** | *Ixodes ricinus* | Castor Bean / Sheep Tick | Lyme Borreliosis, Tick-Borne Encephalitis | 🔴 Critical |
| **Tick** | *Ixodes scapularis* | Blacklegged / Deer Tick | Lyme Disease (*Borrelia burgdorferi*), Babesiosis | 🔴 Critical |
| **Tick** | *Rhipicephalus sanguineus*| Brown Dog Tick | Rocky Mountain Spotted Fever, Canine Ehrlichiosis | 🟠 High |
| **Flea** | *Ctenocephalides canis* | Dog Flea | Dipylidium caninum (Tapeworm) | 🟡 Moderate |
| **Flea** | *Ctenocephalides felis* | Cat Flea | *Bartonella henselae* (Cat Scratch), Murine Typhus | 🟡 Moderate |
| **Louse** | *Pediculus humanus capitis* | Head Louse | Ectoparasite (Pediculosis capitis) | 🟢 Low |
| **Louse** | *Pediculus humanus corporis*| Body Louse | Epidemic Typhus, Trench Fever, Relapsing Fever | 🔴 Severe |
| **Cimicid** | *Cimex lectularius* | Common Bed Bug | Allergic dermatitis, psychological distress | 🟢 Low |

---

## 📊 Experimental Results & Benchmarks

In the research pipeline ([Insect_Model.ipynb](Insect_Model.ipynb)), a baseline 4-stage convolutional neural network trained from scratch was benchmarked against transfer learning architectures:

### 1. Model Performance Comparison

| Model Architecture | Test Accuracy | Precision | Recall | Macro F1 | Epochs | Training Strategy |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Custom CNN (Baseline)** | **85.42%** | **79.11%** | **89.02%** | **81.68%** | 20 | Trained from scratch with class weighting |
| **MobileNetV2** | **90.03%** | **84.37%** | **89.29%** | **86.03%** | 25 | Transfer Learning + Fine-tuning |
| **EfficientNetB0** | **89.61%** | **82.14%** | **90.12%** | **84.21%** | 25 | Transfer Learning + Fine-tuning |

### 2. Dataset Scaling Sensitivity Study
To evaluate model performance in low-data public health regimes, training datasets were sampled at **25%**, **50%**, and **100%**:

```text
Dataset Size | Baseline CNN | MobileNetV2 | EfficientNetB0
--------------------------------------------------------
   25%       |    32.10%    |   87.71%    |     94.67%
   50%       |    55.48%    |   89.62%    |     96.53%
  100%       |    71.50%    |   93.70%    |     97.02%
```
*Takeaway: Transfer learning architectures exhibit exceptional generalization under extreme sample scarcity, whereas training from scratch requires full data saturation.*

---

## 🏗️ Neural Network Architecture

The baseline CNN employs four convolutional blocks followed by global pooling and regularization:

```text
Input (224 × 224 × 3)
   │
Data Augmentation (Flip, Rotation, Zoom)
   │
Rescaling (1/255.0)
   │
Conv2D (32 filters, 3×3, ReLU, Same) ──► BatchNorm ──► MaxPool2D (2×2)
   │
Conv2D (64 filters, 3×3, ReLU, Same) ──► BatchNorm ──► MaxPool2D (2×2)
   │
Conv2D (128 filters, 3×3, ReLU, Same) ──► BatchNorm ──► MaxPool2D (2×2)
   │
Conv2D (256 filters, 3×3, ReLU, Same) ──► BatchNorm ──► MaxPool2D (2×2)
   │
GlobalAveragePooling2D
   │
Dense (256, ReLU) ──► Dropout (0.5)
   │
Dense (16, Softmax) ──► Predicted Probabilities
```

---

## 🚀 Web Application Features

* **Resilient Model Streaming:** Model weights (`trained_model.keras`, 31.4 MB) are fetched on first boot from **GitHub Releases CDN** with a streaming progress bar, preventing cloud IP blocks and quota limits.
* **Top-K Confidence Distribution:** Interactive probability chart displaying top candidate species.
* **Medical Entomology Dossier:** Automated retrieval of taxonomic hierarchy, clinical pathogen associations, and ecological breeding habitats.
* **Sample Presets:** Instant 1-click evaluation with bundled test images.
* **Custom Model Injection:** Sidebar upload widget for evaluating alternative `.keras` checkpoints.

---

## 💻 Local Setup & Installation

### Prerequisites
* Python 3.11 or 3.12 (64-bit)
* Git

### Step-by-Step

```powershell
# 1. Clone the repository
git clone https://github.com/maduekedickson/insect-cnn-baseline.git
cd insect-cnn-baseline

# 2. Create and activate a virtual environment
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1   # On Linux/macOS: source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the application
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ☁️ Streamlit Community Cloud Deployment

1. Fork or push changes to your GitHub repository: `https://github.com/maduekedickson/insect-cnn-baseline`.
2. Navigate to [share.streamlit.io](https://share.streamlit.io/).
3. Connect your repository:
   * **Repository:** `maduekedickson/insect-cnn-baseline`
   * **Branch:** `main`
   * **Main file path:** `app.py`
4. Deploy! The application will automatically stream `trained_model.keras` from GitHub Releases.

---

## 📁 Repository Structure

```text
insect-cnn-baseline/
├── .devcontainer/
│   └── devcontainer.json      # VS Code / Codespaces environment specification
├── sample_images/             # Preloaded sample images for 1-click testing
│   ├── sample_aedes_aegypti.jpg
│   ├── sample_anopheles_stephensi.jpg
│   └── sample_ixodes_scapularis.jpg
├── src/
│   ├── config.py              # Class labels, medical vector metadata & URLs
│   ├── model_loader.py        # Resilient streaming downloader & Keras caching
│   └── utils.py               # Preprocessing (224x224) & top-k ranking logic
├── app.py                     # Streamlit multi-tab production web application
├── Insect_Model.ipynb         # Full research pipeline (CNN, MobileNet, EfficientNet)
├── requirements.txt           # Production dependencies
├── .gitignore                 # Prevents large binary model tracking in git
└── README.md                  # Comprehensive project documentation
```

---

## 📄 License & Attribution

This project is open-source under the [MIT License](LICENSE). Built for academic research and vector surveillance applications.
