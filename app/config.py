import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Project Root
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Models
BINARY_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "mobilenetv3_fine_tune_binary.keras")
MODEL_A_PATH = os.path.join(PROJECT_ROOT, "models", "efficientnetv2m_finetuned.keras")

# Class Names
BINARY_CLASS_NAMES = ["not_skin", "skin"]

STAGE2_CLASS_NAMES = [
    "Bacterial",
    "Psoriasis",
    "eczema",
    "fungal",
    "normal",
    "parasitic infection",
    "viral"
]

# Database path
DB_PATH = os.path.join(BASE_DIR, "data", "disease_info.db")

# Upload settings
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"png","jpg","jpeg","bmp","gif","tiff","webp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB