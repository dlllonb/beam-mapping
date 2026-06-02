"""
Shared constants and data paths for the GRASP beam map analysis suite.

All paths are relative to the beam-mapping/ repository root.
"""

from pathlib import Path

# Directory containing this file → beam-mapping/core/
_CORE_DIR = Path(__file__).parent
# beam-mapping/ root
PLOT_DIR = _CORE_DIR.parent

DATA_DIR = PLOT_DIR / "data"
HORN_LAYOUT_FILE = DATA_DIR / "horn_layout" / "horn_layout_331.dat"
COSMO_MODELS_DIR = DATA_DIR / "cosmological_models"

# Instrument parameters
FREQ_GHZ = 150.0
LAMBDA_MM = 2.0           # c / f = 300 mm·GHz / 150 GHz
EFL_MM = 1200.0           # effective focal length
HORN_SPACING_MM = 15.0    # center-to-center horn pitch
FNUMBER = 2.4             # f/# = EFL / primary diameter
PRIMARY_DIAM_MM = EFL_MM / FNUMBER

# Degrees-to-radians conversion
import math
DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi
RAD2ARCMIN = RAD2DEG * 60.0

# Available tensor-to-scalar ratio models
R_VALUES = [0.001, 0.01, 0.05, 0.1, 0.2]
