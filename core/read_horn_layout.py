"""
Read a GRASP horn-layout .dat file and compute sky-plane offsets.

The .dat file has 5 whitespace-separated columns:
    name  x_mm  y_mm  z_mm  rot_deg

Sky offsets are computed from focal-plane (x, y) positions using the
small-angle approximation:
    theta = r / EFL          (radians, r = sqrt(x^2+y^2))
    phi   = atan2(y, x)
    phi_x = theta * cos(phi) (degrees)
    phi_y = theta * sin(phi) (degrees)

Usage
-----
from core.read_horn_layout import read_horn_layout, compute_sky_offsets
df = read_horn_layout("data/horn_layout/horn_layout_331.dat")
df = compute_sky_offsets(df, efl_mm=1200.0)
"""

import numpy as np
import pandas as pd
from pathlib import Path


def read_horn_layout(dat_path) -> pd.DataFrame:
    """Load horn positions from an ASCII .dat file.

    Returns a DataFrame with columns:
        name, x_mm, y_mm, z_mm, rot_deg
    """
    dat_path = Path(dat_path)
    df = pd.read_csv(
        dat_path,
        sep=r"\s+",
        header=None,
        names=["name", "x_mm", "y_mm", "z_mm", "rot_deg"],
        dtype={"name": str, "x_mm": float, "y_mm": float,
               "z_mm": float, "rot_deg": float},
    )
    # Strip any hyphens from names (e.g. "331-F1" → "331F1") to match GRD filenames
    df["name"] = df["name"].str.replace("-", "", regex=False)
    return df


def compute_sky_offsets(df: pd.DataFrame, efl_mm: float = 1200.0) -> pd.DataFrame:
    """Add phi_x_deg and phi_y_deg columns to a horn-layout DataFrame.

    Parameters
    ----------
    df : DataFrame from read_horn_layout
    efl_mm : effective focal length in mm

    Returns the same DataFrame (in-place modification + return for chaining).
    """
    r = np.sqrt(df["x_mm"] ** 2 + df["y_mm"] ** 2)
    theta = r / efl_mm  # radians
    phi = np.arctan2(df["y_mm"], df["x_mm"])  # radians

    df["phi_x_deg"] = np.degrees(theta * np.cos(phi))
    df["phi_y_deg"] = np.degrees(theta * np.sin(phi))
    return df


def load_layout(dat_path=None, efl_mm: float = 1200.0) -> pd.DataFrame:
    """Convenience: load and annotate with sky offsets in one call."""
    if dat_path is None:
        from core.config import HORN_LAYOUT_FILE
        dat_path = HORN_LAYOUT_FILE
    df = read_horn_layout(dat_path)
    return compute_sky_offsets(df, efl_mm=efl_mm)
