from __future__ import annotations

import sys
from pathlib import Path

# Ensure the src/ directory is on sys.path so we can import pses_chatbot
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pses_chatbot.ui.app import run_app  # type: ignore


if __name__ == "__main__":
    run_app()
