"""
Beam characterisation utilities.

Key functions
-------------
characterize_beam(beam_map, u, v)
    → dict with fwhm_major_arcmin, fwhm_minor_arcmin, orientation_deg,
              center_u, center_v, eccentricity, ellipticity, gain_dBi

find_hm_contour(beam_map, u, v, level=None)
    → ndarray shape (2, N) of contour crossing coordinates

mask_beam(u, v, beam_map, cutoff_rad)
    → bool ndarray, True where distance from peak ≥ cutoff_rad

power_beam(field_map)
    → real-valued power (|E|^2 summed over components)
"""

import numpy as np
from core.config import RAD2ARCMIN


def power_beam(field_map: np.ndarray) -> np.ndarray:
    """Convert complex E-field map(s) to power (|E|²).

    Parameters
    ----------
    field_map : (..., 2) complex ndarray
        Last axis holds the two field components.

    Returns
    -------
    ndarray without last axis, dtype float64
    """
    return (np.abs(field_map) ** 2).sum(axis=-1)


def find_hm_contour(
    beam_map: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    level: float = None,
) -> np.ndarray:
    """Find contour crossings at *level* via linear interpolation.

    Scans every column and every row looking for sign changes around *level*,
    and interpolates to sub-pixel precision.  Matches the algorithm in
    IDL find_hm.pro.

    Parameters
    ----------
    beam_map : 2D float ndarray, shape (nu, nv)
    u, v : 1D coordinate arrays
    level : float, default = half-maximum of beam_map

    Returns
    -------
    points : ndarray, shape (2, N)
        points[0] = u coordinates, points[1] = v coordinates
    """
    if level is None:
        level = beam_map.max() * 0.5

    u_pts, v_pts = [], []

    # Scan columns (u fixed, vary v)
    for i, ui in enumerate(u):
        col = beam_map[i, :]
        for j in range(len(v) - 1):
            a, b = col[j] - level, col[j + 1] - level
            if a * b < 0:
                frac = -a / (b - a)
                v_pts.append(v[j] + frac * (v[j + 1] - v[j]))
                u_pts.append(ui)

    # Scan rows (v fixed, vary u)
    for j, vj in enumerate(v):
        row = beam_map[:, j]
        for i in range(len(u) - 1):
            a, b = row[i] - level, row[i + 1] - level
            if a * b < 0:
                frac = -a / (b - a)
                u_pts.append(u[i] + frac * (u[i + 1] - u[i]))
                v_pts.append(vj)

    if len(u_pts) == 0:
        return np.empty((2, 0))

    return np.array([u_pts, v_pts])


def characterize_beam(
    beam_map: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> dict:
    """Characterise a 2D beam map.

    Finds the half-maximum contour and analyses it using eigendecomposition
    of the 2×2 spatial covariance matrix, mirroring IDL characterize_beam.pro.

    Parameters
    ----------
    beam_map : 2D float ndarray, shape (nu, nv)
    u, v : 1D coordinate arrays in radians

    Returns
    -------
    dict with keys:
        fwhm_major_arcmin, fwhm_minor_arcmin  — FWHM along principal axes
        orientation_deg   — angle of major axis from +u toward +v (degrees)
        center_u, center_v — peak position in radians
        eccentricity      — sqrt(1 - (minor/major)^2)
        ellipticity       — (major - minor) / (major + minor)
        gain_dBi          — 10*log10(peak / (4*pi solid angle) )
                            here simplified as 10*log10(peak)
    """
    peak_val = beam_map.max()
    peak_idx = np.unravel_index(beam_map.argmax(), beam_map.shape)
    center_u = u[peak_idx[0]]
    center_v = v[peak_idx[1]]

    level = peak_val * 0.5
    pts = find_hm_contour(beam_map, u, v, level=level)

    if pts.shape[1] < 4:
        # Not enough contour points — return degenerate result
        return dict(
            fwhm_major_arcmin=np.nan, fwhm_minor_arcmin=np.nan,
            orientation_deg=0.0, center_u=center_u, center_v=center_v,
            eccentricity=np.nan, ellipticity=np.nan, gain_dBi=np.nan,
        )

    pu, pv = pts[0], pts[1]

    # 2×2 covariance matrix of contour point cloud
    cov = np.cov(pu, pv)  # shape (2,2)

    # Eigendecomposition: eigenvalues give variance along principal axes
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh returns ascending eigenvalues; major axis has larger eigenvalue
    idx_major = np.argmax(eigvals)
    idx_minor = 1 - idx_major

    # Rotate contour points into principal-axis frame
    principal = eigvecs[:, idx_major]  # unit vector of major axis
    angle = np.arctan2(principal[1], principal[0])
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    pu_rot = cos_a * pu + sin_a * pv
    pv_rot = -sin_a * pu + cos_a * pv

    fwhm_major_rad = pu_rot.max() - pu_rot.min()
    fwhm_minor_rad = pv_rot.max() - pv_rot.min()

    fwhm_major = fwhm_major_rad * RAD2ARCMIN
    fwhm_minor = fwhm_minor_rad * RAD2ARCMIN

    if fwhm_major == 0:
        eccentricity = np.nan
        ellipticity = np.nan
    else:
        ratio = fwhm_minor / fwhm_major
        eccentricity = np.sqrt(1.0 - ratio ** 2)
        ellipticity = (fwhm_major - fwhm_minor) / (fwhm_major + fwhm_minor)

    orientation_deg = np.degrees(angle)

    # Gain: treat peak value as directive gain proxy; 10*log10 of normalised peak
    gain_dBi = 10.0 * np.log10(peak_val) if peak_val > 0 else np.nan

    return dict(
        fwhm_major_arcmin=fwhm_major,
        fwhm_minor_arcmin=fwhm_minor,
        orientation_deg=orientation_deg,
        center_u=center_u,
        center_v=center_v,
        eccentricity=eccentricity,
        ellipticity=ellipticity,
        gain_dBi=gain_dBi,
    )


def mask_beam(
    u: np.ndarray,
    v: np.ndarray,
    beam_map: np.ndarray,
    cutoff_rad: float,
) -> np.ndarray:
    """Return a boolean mask that is True outside *cutoff_rad* from the peak.

    Parameters
    ----------
    u, v : 1D coordinate arrays in radians
    beam_map : 2D float ndarray, shape (nu, nv)
    cutoff_rad : radius threshold in radians

    Returns
    -------
    mask : bool ndarray, shape (nu, nv)
        True where radius >= cutoff_rad (i.e. the sidelobe / far-field region).
    """
    peak_idx = np.unravel_index(beam_map.argmax(), beam_map.shape)
    uu, vv = np.meshgrid(u, v, indexing="ij")  # (nu, nv)
    radius = np.sqrt((uu - u[peak_idx[0]]) ** 2 + (vv - v[peak_idx[1]]) ** 2)
    return radius >= cutoff_rad
