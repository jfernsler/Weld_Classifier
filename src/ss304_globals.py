from pathlib import Path
import os

# Globals
SCRIPT_PATH = Path(__file__).absolute()
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
CSV_DIR = os.path.join(SCRIPT_DIR, '..', 'csv')
CHART_DIR = os.path.join(SCRIPT_DIR, '..', 'charts')
MODEL_DIR = os.path.join(SCRIPT_DIR, '..', 'models')
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
DATA_SRC = 'ss304'
DATA_SRC_REDUCED = 'ss304_reduced'
