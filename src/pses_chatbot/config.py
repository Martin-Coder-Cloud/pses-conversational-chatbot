from __future__ import annotations
from pathlib import Path

# Root of the project (repo root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories (we'll refine these as we go)
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "pses_results"
METADATA_DIR = DATA_DIR / "metadata"

# App name and version
APP_NAME = "PSES Conversational Analytics Chatbot"
APP_VERSION = "0.1.0"
