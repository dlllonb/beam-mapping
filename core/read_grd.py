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

Coordinate units
----------------
GRASP .grd files embed an ICOORD flag in the header:
    ICOORD=1  direction cosines (u, v) — dimensionless / effectively radians
    ICOORD=2  spherical angles (theta, phi) — DEGREES
    ICOORD=3+ other angle conventions        — DEGREES

By default the parser reads ICOORD and converts degree-based extents to
radians automatically.  Use ``force_units="rad"`` or ``force_units="deg"``
to override when the file header is non-standard.
"""

import numpy as np

import math
_DEG2RAD = math.pi / 180.0


def read_grd(filename: str, force_units: str = "auto"):
    """Parse a TICRA GRASP .grd file.

    Parameters
    ----------
    filename : str or Path
    force_units : {"auto", "rad", "deg"}
        Units of the grid coordinate extents.
        ``"auto"`` (default) reads ICOORD from the file header:
            ICOORD=1 → direction cosines (radians, no conversion needed)
            ICOORD≥2 → spherical/Ludwig angles in degrees → converted to radians
        ``"rad"``  forces the extents to be treated as radians regardless.
        ``"deg"``  forces the extents to be treated as degrees and converts.
    """
    with open(filename, "r") as fh:
        lines = fh.readlines()

    # Find the "++++"-delimited header boundary
    delim_idx = next(i for i, ln in enumerate(lines) if ln.strip() == "++++")

    # Lines after the delimiter hold grid metadata, then data
    meta_lines = lines[delim_idx + 1:]

    # Line 0: "1"              (number of field sets)
    # Line 1: "ICOMP NCOMP IGRID ICOORD"
    # Line 2: "0  0"           (polarisation reference — unused)
    # Line 3: "u_min v_min u_max v_max"   (grid extent)
    # Line 4: "nu  nv  0"      (grid dimensions)
    # Lines 5+: data

    # Parse ICOORD to determine coordinate units
    header_tokens = meta_lines[1].split()
    icoord = int(header_tokens[3]) if len(header_tokens) >= 4 else 1

    # Parse grid extents
    extent_tokens = meta_lines[3].split()
    u_min, v_min, u_max, v_max = [float(t) for t in extent_tokens]

    # Convert to radians if needed
    if force_units == "deg" or (force_units == "auto" and icoord >= 2):
        u_min *= _DEG2RAD
        v_min *= _DEG2RAD
        u_max *= _DEG2RAD
        v_max *= _DEG2RAD
        if force_units == "auto":
            print(f"  [read_grd] ICOORD={icoord}: interpreting grid extents as "
                  f"degrees and converting to radians.")
    elif force_units not in ("rad", "auto"):
        raise ValueError(f"force_units must be 'auto', 'rad', or 'deg', got {force_units!r}")

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
