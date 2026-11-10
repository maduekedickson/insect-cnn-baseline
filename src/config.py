"""
Project Configuration and Species Metadata
Vector-Borne Disease and Medical Entomology Information
"""

import os

# ==========================================
# MODEL CONFIGURATION
# ==========================================
IMG_SIZE = (224, 224)
NUM_CLASSES = 16

# Primary release download URL (GitHub Releases asset)
DEFAULT_RELEASE_URL = os.getenv(
    "MODEL_URL",
    "https://github.com/maduekedickson/insect-cnn-baseline/releases/download/deep_learning/trained_model.keras"
)

# Backup Google Drive ID (legacy fallback)
GDRIVE_FILE_ID = os.getenv("GDRIVE_FILE_ID", "1iZb55Lq1jf4h48OpBL6irrv0LXKOKHVV")

# Local cache path
LOCAL_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cnn_best.keras")

# ==========================================
# CLASS LABELS (16 Species)
# ==========================================
CLASS_NAMES = [
    "Aedes_aegypti",
    "Aedes_albopictus",
    "Aedes_vexans",
    "Amblyomma_americanum",
    "Anopheles_stephensi",
    "Anopheles_tessellatus",
    "Cimex_lectularius",
    "Ctenocephalides_canis",
    "Ctenocephalides_felis",
    "Culex_quinquefasciatus",
    "Culex_vishnui",
    "Ixodes_ricinus",
    "Ixodes_scapularis",
    "Pediculus_humanus_capitis",
    "Pediculus_humanus_corporis",
    "Rhipicephalus_sanguineus"
]

# ==========================================
# ENTOMOLOGY & MEDICAL VECTOR DATABASE
# ==========================================
SPECIES_METADATA = {
    "Aedes_aegypti": {
        "common_name": "Yellow Fever Mosquito",
        "category": "Mosquito (Culicidae)",
        "vector_status": "High-Risk Primary Vector",
        "pathogens": ["Dengue Virus", "Zika Virus", "Chikungunya Virus", "Yellow Fever Virus"],
        "habitat": "Urban & peri-urban environments, artificial water containers.",
        "risk_level": "Severe"
    },
    "Aedes_albopictus": {
        "common_name": "Asian Tiger Mosquito",
        "category": "Mosquito (Culicidae)",
        "vector_status": "High-Risk Vector",
        "pathogens": ["Dengue Virus", "Chikungunya Virus", "Zika Virus", "Dirofilaria immitis"],
        "habitat": "Woodland edges, gardens, urban tires and water collection sites.",
        "risk_level": "Severe"
    },
    "Aedes_vexans": {
        "common_name": "Inland Floodwater Mosquito",
        "category": "Mosquito (Culicidae)",
        "vector_status": "Secondary Vector",
        "pathogens": ["Rift Valley Fever", "West Nile Virus", "Eastern Equine Encephalitis"],
        "habitat": "Temporary pools, flooded meadows, irrigation pools.",
        "risk_level": "Moderate"
    },
    "Amblyomma_americanum": {
        "common_name": "Lone Star Tick",
        "category": "Tick (Ixodidae)",
        "vector_status": "Primary Vector / Allergenic",
        "pathogens": ["Ehrlichia chaffeensis (Ehrlichiosis)", "Heartland Virus", "Bourbon Virus", "Alpha-Gal Syndrome (Red Meat Allergy)"],
        "habitat": "Dense woodlands, thick brush, ecotones.",
        "risk_level": "High"
    },
    "Anopheles_stephensi": {
        "common_name": "Indo-Pakistan Malaria Mosquito",
        "category": "Mosquito (Culicidae)",
        "vector_status": "High-Risk Primary Vector",
        "pathogens": ["Plasmodium falciparum (Malaria)", "Plasmodium vivax (Malaria)"],
        "habitat": "Urban water tanks, construction sites, cisterns.",
        "risk_level": "Critical"
    },
    "Anopheles_tessellatus": {
        "common_name": "Tessellated Mosquito",
        "category": "Mosquito (Culicidae)",
        "vector_status": "Secondary Vector",
        "pathogens": ["Plasmodium vivax (Secondary Malaria Vector)"],
        "habitat": "Shaded streams, seepage pools, river margins.",
        "risk_level": "Moderate"
    },
    "Cimex_lectularius": {
        "common_name": "Common Bed Bug",
        "category": "True Bug (Cimicidae)",
        "vector_status": "Nuisance / Allergic Pest",
        "pathogens": ["No known systemic disease transmission; causes allergic dermatitis & psychological distress"],
        "habitat": "Mattresses, furniture crevices, bedroom wall cracks.",
        "risk_level": "Low"
    },
    "Ctenocephalides_canis": {
        "common_name": "Dog Flea",
        "category": "Flea (Pulicidae)",
        "vector_status": "Parasitic Vector",
        "pathogens": ["Dipylidium caninum (Dog Tapeworm)", "Rickettsia felis"],
        "habitat": "Domestic dogs, kennels, animal bedding.",
        "risk_level": "Moderate"
    },
    "Ctenocephalides_felis": {
        "common_name": "Cat Flea",
        "category": "Flea (Pulicidae)",
        "vector_status": "Parasitic / Zoonotic Vector",
        "pathogens": ["Bartonella henselae (Cat Scratch Disease)", "Rickettsia typhi (Murine Typhus)", "Dipylidium caninum"],
        "habitat": "Domestic cats, dogs, carpets, indoor upholstery.",
        "risk_level": "Moderate"
    },
    "Culex_quinquefasciatus": {
        "common_name": "Southern House Mosquito",
        "category": "Mosquito (Culicidae)",
        "vector_status": "High-Risk Primary Vector",
        "pathogens": ["Wuchereria bancrofti (Lymphatic Filariasis)", "West Nile Virus", "St. Louis Encephalitis Virus"],
        "habitat": "Organically polluted stagnant water, drains, septic pits.",
        "risk_level": "Severe"
    },
    "Culex_vishnui": {
        "common_name": "Vishnui Mosquito",
        "category": "Mosquito (Culicidae)",
        "vector_status": "High-Risk Vector",
        "pathogens": ["Japanese Encephalitis Virus (JEV)", "West Nile Virus"],
        "habitat": "Rice paddies, marshes, rural ponds.",
        "risk_level": "Severe"
    },
    "Ixodes_ricinus": {
        "common_name": "Castor Bean / Sheep Tick",
        "category": "Tick (Ixodidae)",
        "vector_status": "High-Risk Primary Vector",
        "pathogens": ["Borrelia burgdorferi (Lyme Disease)", "Tick-Borne Encephalitis Virus (TBEV)", "Anaplasma phagocytophilum"],
        "habitat": "Deciduous woodlands, tall grass, European temperate zones.",
        "risk_level": "Critical"
    },
    "Ixodes_scapularis": {
        "common_name": "Blacklegged / Deer Tick",
        "category": "Tick (Ixodidae)",
        "vector_status": "High-Risk Primary Vector",
        "pathogens": ["Borrelia burgdorferi (Lyme Disease)", "Babesia microti (Babesiosis)", "Powassan Virus"],
        "habitat": "Leaf litter, deciduous forests, coastal scrub in North America.",
        "risk_level": "Critical"
    },
    "Pediculus_humanus_capitis": {
        "common_name": "Head Louse",
        "category": "Louse (Pediculidae)",
        "vector_status": "Obligate Ectoparasite",
        "pathogens": ["Non-vector; causes pediculosis capitis, intense scalp itching, secondary bacterial infection"],
        "habitat": "Human scalp and hair shafts.",
        "risk_level": "Low"
    },
    "Pediculus_humanus_corporis": {
        "common_name": "Body Louse",
        "category": "Louse (Pediculidae)",
        "vector_status": "High-Risk Primary Vector",
        "pathogens": ["Rickettsia prowazekii (Epidemic Typhus)", "Bartonella quintana (Trench Fever)", "Borrelia recurrentis (Relapsing Fever)"],
        "habitat": "Clothing seams of individuals in overcrowded/refugee conditions.",
        "risk_level": "Severe"
    },
    "Rhipicephalus_sanguineus": {
        "common_name": "Brown Dog Tick",
        "category": "Tick (Ixodidae)",
        "vector_status": "Zoonotic Vector",
        "pathogens": ["Rickettsia rickettsii (Rocky Mountain Spotted Fever)", "Ehrlichia canis", "Babesia canis"],
        "habitat": "Kennels, dog homes, domestic wall crevices.",
        "risk_level": "High"
    }
}

# ==========================================
# BENCHMARK COMPARISON METRICS (From Notebook)
# ==========================================
BENCHMARK_RESULTS = {
    "Overall_Models": [
        {"Model": "Baseline CNN (From Scratch)", "Accuracy": 0.8542, "Precision": 0.7911, "Recall": 0.8902, "Macro_F1": 0.8168, "Epochs": 20, "Params": "~1.2M"},
        {"Model": "MobileNetV2 (Transfer Learning)", "Accuracy": 0.9003, "Precision": 0.8437, "Recall": 0.8929, "Macro_F1": 0.8603, "Epochs": 25, "Params": "~2.4M"},
        {"Model": "EfficientNetB0 (Transfer Learning)", "Accuracy": 0.8961, "Precision": 0.8214, "Recall": 0.9012, "Macro_F1": 0.8421, "Epochs": 25, "Params": "~4.2M"}
    ],
    "Dataset_Scaling": [
        {"Dataset_Size": "25%", "CNN": 0.3210, "MobileNetV2": 0.8771, "EfficientNetB0": 0.9467},
        {"Dataset_Size": "50%", "CNN": 0.5548, "MobileNetV2": 0.8962, "EfficientNetB0": 0.9653},
        {"Dataset_Size": "100%", "CNN": 0.7150, "MobileNetV2": 0.9370, "EfficientNetB0": 0.9702}
    ]
}
