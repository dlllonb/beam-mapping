"""
Parser for TICRA-EM-FIELD-V0.1 (.grd) files produced by GRASP.

Usage
-----
u, v, field_maps = read_grd("path/to/beam_map.grd")

Returns
-------
u : ndarray, shape (nu,)
    U-axis coordinates in radians (linearly spaced).
v : ndarray, shape (nv,)
    V-axis coordinates in radians (linearly spaced).
field_maps : ndarray, shape (nu, nv, 2), dtype complex128
    field_maps[:, :, 0] = first field component  (E_x or E_co)
    field_maps[:, :, 1] = second field component (E_y or E_cx)
    Column-major ordering: u index varies fastest in the raw data.
"""

import re
import numpy as np


def read_grd(filename: str):
    """Parse a TICRA GRASP .grd file.

    Parameters
    ----------
    filename : str or Path
    """
    with open(filename, "r") as fh:
        lines = fh.readlines()

    # Find the "++++"-delimited header boundary
    delim_idx = next(i for i, ln in enumerate(lines) if ln.strip() == "++++")

    # Lines after the delimiter hold grid metadata, then data
    meta_lines = lines[delim_idx + 1:]

    # Skip the field-count line (always "1" for single frequency)
    # Line 0: "1"  (number of field sets)
    # Line 1: "1  3  2  1" (ICOMP NCOMP IGRID ICOORD — we need NCOMP=2 always)
    # Line 2: "0  0" (unused polarization reference)
    # Line 3: "u_min  v_min  u_max  v_max" (grid extent in radians)
    # Line 4: "nu  nv  0" (grid dimensions)
    # Lines 5+: data

    # Parse grid extents
    extent_tokens = meta_lines[3].split()
    u_min, v_min, u_max, v_max = [float(t) for t in extent_tokens]

    # Parse grid dimensions
    dim_tokens = meta_lines[4].split()
    nu, nv = int(dim_tokens[0]), int(dim_tokens[1])

    # Parse data block: each row is "Re1 Im1 Re2 Im2"
    data_lines = meta_lines[5:]
    values = []
    for ln in data_lines:
        ln = ln.strip()
        if not ln:
            continue
        values.extend(float(x) for x in ln.split())

    values = np.array(values)
    # Total points = nu * nv, each with 4 floats (Re1 Im1 Re2 Im2)
    n_pts = nu * nv
    values = values[: n_pts * 4].reshape(n_pts, 4)

    # Build complex arrays; data is column-major (u varies fastest)
    e1 = (values[:, 0] + 1j * values[:, 1]).reshape(nu, nv, order="F")
    e2 = (values[:, 2] + 1j * values[:, 3]).reshape(nu, nv, order="F")

    field_maps = np.stack([e1, e2], axis=-1)  # (nu, nv, 2)

    u = np.linspace(u_min, u_max, nu)
    v = np.linspace(v_min, v_max, nv)

    return u, v, field_maps
