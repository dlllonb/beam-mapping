"""
analyze_wiregrid_mueller_nonideal.py

Quantitative non-ideal polarization diagnostics for wire-grid beam maps.

Compares a no-grid (feed-only) GRASP simulation with a wire-grid-inserted
simulation, both processed through the project's standard Mueller pipeline.
Estimates the effective partial-polarizer behaviour of the grid and maps the
residual non-ideal effects: I->U leakage, spatially-varying polarization angle,
and deviation from a simple uniform partial-polarizer model.

Convention
----------
Mueller element M[r, c] maps Stokes-c input to Stokes-r output:
    S_out = M @ S_in   (standard Mueller calculus)
Stokes index: 0 = T (intensity), 1 = Q, 2 = U, 3 = V.
Maps are normalised so integral of M[0,0](u,v) du dv = 1 (see core/mueller.py).
The stored integrated Mueller matrix therefore has M[0,0] = 1 by construction.

Usage
-----
python scripts/analyze_wiregrid_mueller_nonideal.py \\
    --nogrid-pol1 data/maps/wiregrid_maps/azel_1deg_offx.grd \\
    --nogrid-pol2 data/maps/wiregrid_maps/azel_1deg_offy.grd \\
    --grid-pol1   data/maps/wiregrid_maps/azel_1deg_densex.grd \\
    --grid-pol2   data/maps/wiregrid_maps/azel_1deg_densey.grd \\
    --outdir      outputs/wiregrid_mueller_nonideal
"""

import argparse
import csv
import json
import sys
from math import atan2, degrees, sqrt, pi
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
from matplotlib.backends.backend_pdf import PdfPages

# ---------------------------------------------------------------------------
# Project imports -- allow running from the repository root or from scripts/
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.read_grd import read_grd
from core.mueller import build_mueller_maps, integrate_element
from core.config import RAD2ARCMIN

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STOKES = ["T", "Q", "U", "V"]
LABELS_4x4 = [[f"M_{r}{c}" for c in STOKES] for r in STOKES]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _extent_arcmin(u, v):
    """Return matplotlib extent in arcmin for imshow."""
    return [u[0] * RAD2ARCMIN, u[-1] * RAD2ARCMIN,
            v[0] * RAD2ARCMIN, v[-1] * RAD2ARCMIN]


def _build_integrated_matrix(mueller, u, v):
    """Integrate all 16 Mueller elements over solid angle -> (4,4) array."""
    return np.array([
        [integrate_element(mueller, r, c, u, v) for c in range(4)]
        for r in range(4)
    ])


def _print_matrix(label, mm, indent=2):
    pad = " " * indent
    print(f"{pad}{label}:")
    for row in mm:
        print(pad + "  " + "  ".join(f"{x:+.6f}" for x in row))


# ---------------------------------------------------------------------------
# Step 1 & 2 -- Load .grd files and build Mueller maps
# ---------------------------------------------------------------------------

def load_case(pol1_path, pol2_path, label, grd_units):
    """Read two .grd files, build normalised Mueller maps and integrated matrix."""
    print(f"\n[{label}] Reading pol1: {pol1_path}")
    u, v, field_pol1 = read_grd(pol1_path, force_units=grd_units)
    print(f"[{label}] Reading pol2: {pol2_path}")
    _, _, field_pol2 = read_grd(pol2_path, force_units=grd_units)

    print(f"[{label}] Building Mueller maps ({u.size}x{v.size} pixels)...")
    mueller = build_mueller_maps(field_pol1, field_pol2, u, v)

    mm = _build_integrated_matrix(mueller, u, v)
    print(f"[{label}] Integrated Mueller matrix (normalised, M[0,0] = 1 by construction):")
    _print_matrix("M_int", mm)

    return u, v, mueller, mm


# ---------------------------------------------------------------------------
# Step 3 -- Comparison printout
# ---------------------------------------------------------------------------

def print_comparison(mm_nogrid, mm_grid):
    print("\n" + "=" * 70)
    print("INTEGRATED MUELLER MATRIX COMPARISON")
    print("=" * 70)
    _print_matrix("No-grid", mm_nogrid)
    _print_matrix("Wire-grid", mm_grid)
    diff = mm_grid - mm_nogrid
    _print_matrix("Difference (grid - nogrid)", diff)

    # Ratio where both are non-negligible
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(mm_nogrid) > 1e-4, mm_grid / mm_nogrid, np.nan)
    _print_matrix("Ratio grid/nogrid (|nogrid| > 1e-4)", ratio)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Step 4 -- Partial-polarizer characterisation
# ---------------------------------------------------------------------------

def partial_polarizer_analysis(mm_grid):
    """Estimate grid polarization selectivity from the integrated Mueller matrix.

    For an ideal partial linear polarizer with power transmittances T_x, T_y:
        M_TT (normalised) = 1
        M_TQ / M_TT = (T_x - T_y) / (T_x + T_y) = p_TQ
        M_QT / M_TT = same value (symmetric for a pure polarizer)
        T_blocked / T_passed = (1 - |p|) / (1 + |p|)

    Returns a dict of diagnostic quantities.
    """
    M_TT = mm_grid[0, 0]
    M_TQ = mm_grid[0, 1]
    M_QT = mm_grid[1, 0]
    M_QQ = mm_grid[1, 1]
    M_UU = mm_grid[2, 2]
    M_VV = mm_grid[3, 3]
    M_UV = mm_grid[2, 3]
    M_VU = mm_grid[3, 2]

    p_TQ = M_TQ / M_TT
    p_QT = M_QT / M_TT

    p_mean = 0.5 * (p_TQ + p_QT)
    abs_p = abs(p_mean)

    # T_blocked / T_passed from partial-polarizer relation
    T_ratio = (1.0 - abs_p) / (1.0 + abs_p) if (1.0 + abs_p) != 0 else np.nan

    # Amplitude of UU/VV block: sqrt(UU^2 + UV^2) = m33 (ideal polarizer)
    m33_uu = sqrt(M_UU**2 + M_UV**2)
    m33_vv = sqrt(M_VV**2 + M_VU**2)

    # Retardation angle in UV plane: angle of the 2x2 block rotation
    # [[M_UU, M_UV], [M_VU, M_VV]] ~ m33 * [[cos(delta), -sin(delta)],
    #                                         [sin(delta),  cos(delta)]]
    delta_deg = degrees(atan2(-M_UV, M_UU))

    results = dict(
        M_TT=float(M_TT),
        M_TQ=float(M_TQ),
        M_QT=float(M_QT),
        M_QQ=float(M_QQ),
        p_TQ=float(p_TQ),
        p_QT=float(p_QT),
        p_mean=float(p_mean),
        abs_p=float(abs_p),
        T_blocked_over_T_passed=float(T_ratio),
        T_blocked_percent=float(T_ratio * 100),
        m33_amplitude_UU_block=float(m33_uu),
        m33_amplitude_VV_block=float(m33_vv),
        retardation_angle_deg=float(delta_deg),
    )

    print("\n" + "=" * 70)
    print("PARTIAL-POLARIZER CHARACTERISATION")
    print("=" * 70)
    print(f"  p_TQ = M_TQ / M_TT          = {p_TQ:+.6f}")
    print(f"  p_QT = M_QT / M_TT          = {p_QT:+.6f}")
    print(f"  |p| (mean of TQ, QT)        = {abs_p:.6f}")
    print(f"  T_blocked / T_passed        = {T_ratio:.6f}  ({T_ratio*100:.2f}%)")
    print(f"  m33 amplitude (UU block)    = {m33_uu:.6f}")
    print(f"  m33 amplitude (VV block)    = {m33_vv:.6f}")
    print(f"  UV-plane retardation angle  = {delta_deg:.2f} deg")
    print("  Interpretation:")
    print(f"    The wire grid blocks {(1-T_ratio)*100:.1f}% of the intensity in the")
    print(f"    blocked polarization, transmitting {T_ratio*100:.1f}% slip-through.")
    print(f"    The |p|={abs_p:.3f} I<->Q coupling converts ~{abs_p*100:.1f}% of")
    print(f"    integrated intensity into apparent Q signal.")
    print("=" * 70)

    return results


# ---------------------------------------------------------------------------
# Step 5 -- Unpolarized-input diagnostics (integrated)
# ---------------------------------------------------------------------------

def unpolarized_input_analysis(mm_grid):
    """First column of M_int: response to unpolarized (T-only) input.

    S_in = (1, 0, 0, 0) -> S_out = M[:,0] = (TT, QT, UT, VT)
    """
    TT = mm_grid[0, 0]
    QT = mm_grid[1, 0]
    UT = mm_grid[2, 0]
    VT = mm_grid[3, 0]

    P_over_I = sqrt(QT**2 + UT**2) / TT if TT != 0 else np.nan
    psi_eff_rad = 0.5 * atan2(UT, QT)
    psi_eff_deg = degrees(psi_eff_rad)

    results = dict(
        TT_int=float(TT),
        QT_int=float(QT),
        UT_int=float(UT),
        VT_int=float(VT),
        QT_over_TT=float(QT / TT) if TT != 0 else np.nan,
        UT_over_TT=float(UT / TT) if TT != 0 else np.nan,
        P_over_I=float(P_over_I),
        psi_eff_deg=float(psi_eff_deg),
    )

    print("\n" + "=" * 70)
    print("UNPOLARIZED-INPUT DIAGNOSTICS (integrated)")
    print("  S_in = (I, 0, 0, 0)  ->  S_out = M[:,0] = (TT, QT, UT, VT)")
    print("=" * 70)
    print(f"  TT_int               = {TT:+.6f}")
    print(f"  QT_int               = {QT:+.6f}")
    print(f"  UT_int               = {UT:+.6f}")
    print(f"  VT_int               = {VT:+.6f}")
    print(f"  QT / TT              = {QT/TT:+.6f}  ({QT/TT*100:.2f}%)")
    print(f"  UT / TT              = {UT/TT:+.6f}  ({UT/TT*100:.4f}%)")
    print(f"  P/I = sqrt(QT^2+UT^2) / TT = {P_over_I:.6f}  ({P_over_I*100:.2f}%)")
    print(f"  Apparent pol. angle  psi = 0.5*atan2(UT, QT) = {psi_eff_deg:.4f} deg")
    print("  (psi=90 deg means Q-axis polarization; psi=0 means +Q axis)")
    print("=" * 70)

    return results


# ---------------------------------------------------------------------------
# Step 6 & 7 -- Beam-map versions of first-column quantities
# ---------------------------------------------------------------------------

def first_column_maps(mueller_grid, u, v, tt_threshold_frac=1e-3):
    """Compute pixel-by-pixel first-column quantities.

    Returns a dict of 2D arrays (nu, nv), with a boolean mask.
    Quantities are only valid inside the mask (True = valid pixel).
    """
    TT = mueller_grid[:, :, 0, 0]
    QT = mueller_grid[:, :, 1, 0]
    UT = mueller_grid[:, :, 2, 0]
    VT = mueller_grid[:, :, 3, 0]

    TT_peak = TT.max()
    mask = TT > tt_threshold_frac * TT_peak  # True = valid (inside beam)

    with np.errstate(divide="ignore", invalid="ignore"):
        QT_over_TT = np.where(mask, QT / TT, np.nan)
        UT_over_TT = np.where(mask, UT / TT, np.nan)
        P_over_I = np.where(mask, np.sqrt(QT**2 + UT**2) / TT, np.nan)
        psi_map = np.where(mask, 0.5 * np.arctan2(UT, QT) * (180.0 / pi), np.nan)

    return dict(
        TT=TT, QT=QT, UT=UT, VT=VT,
        QT_over_TT=QT_over_TT,
        UT_over_TT=UT_over_TT,
        P_over_I=P_over_I,
        psi_deg=psi_map,
        mask=mask,
        TT_peak=float(TT_peak),
        tt_threshold_frac=tt_threshold_frac,
    )


def beam_weighted_angle_stats(maps_dict):
    """Beam-weighted statistics of the polarization-angle map inside the mask.

    Polarization angles have a 180 deg ambiguity (psi and psi+180 deg are identical),
    so naive arithmetic mean/RMS of psi fail when angles straddle +/-90 deg.
    We use circular statistics via the double-angle trick:

        mean_psi = 0.5 * atan2( sum(w*sin(2*psi)), sum(w*cos(2*psi)) )
        circular_std = 0.5 * sqrt( -2 * log(R) )  where R is the mean resultant length

    This correctly handles the +/-90 deg boundary that arises when QT < 0, UT ~ 0
    (both +90 deg and -90 deg represent the same polarization state).

    For angle deviations we also compute statistics of Delta_psi = psi - psi_mean,
    wrapping each pixel's deviation into (-90 deg, +90 deg].  These deviation statistics
    are the non-ideal diagnostic: how much does the apparent angle vary spatially?

    Weights = TT(u,v) (total intensity beam pattern).
    """
    TT = maps_dict["TT"]
    psi_deg = maps_dict["psi_deg"]
    mask = maps_dict["mask"]

    w = TT[mask]
    psi_m = psi_deg[mask]  # degrees
    psi_r = np.deg2rad(psi_m)  # radians, for double-angle trig

    w_sum = w.sum()

    # Circular (weighted) mean using double-angle representation
    if w_sum > 0:
        sin2 = np.sum(w * np.sin(2 * psi_r)) / w_sum
        cos2 = np.sum(w * np.cos(2 * psi_r)) / w_sum
        mean_psi_rad = 0.5 * atan2(sin2, cos2)
        mean_psi = float(degrees(mean_psi_rad))
        # Mean resultant length R in [0, 1]; circular std = 0.5 * sqrt(-2 ln R)
        R = sqrt(sin2**2 + cos2**2)
        R = min(R, 1.0 - 1e-12)  # guard against log(0)
        circ_std_deg = float(0.5 * degrees(sqrt(-2.0 * np.log(R))))
    else:
        mean_psi = np.nan
        circ_std_deg = np.nan
        mean_psi_rad = 0.0

    # Wrap individual angles to deviation from the circular mean, in (-90, 90]
    delta_rad = np.mod(psi_r - mean_psi_rad + pi / 2, pi) - pi / 2
    delta_deg_vals = np.degrees(delta_rad)

    # Weighted RMS of the deviation (in degrees)
    if w_sum > 0:
        rms_delta = float(sqrt(np.sum(w * delta_deg_vals**2) / w_sum))
        min_delta = float(delta_deg_vals.min())
        max_delta = float(delta_deg_vals.max())
    else:
        rms_delta = np.nan
        min_delta = np.nan
        max_delta = np.nan

    # P/I-weighted mean (also circular)
    P = maps_dict["P_over_I"]
    wp = P[mask]
    wp_sum = wp.sum()
    if wp_sum > 0:
        s2p = np.sum(wp * np.sin(2 * psi_r)) / wp_sum
        c2p = np.sum(wp * np.cos(2 * psi_r)) / wp_sum
        mean_psi_Pw = float(degrees(0.5 * atan2(s2p, c2p)))
    else:
        mean_psi_Pw = np.nan

    results = dict(
        weight_used="TT(u,v)",
        n_valid_pixels=int(mask.sum()),
        circular_mean_psi_deg=mean_psi,
        circular_std_psi_deg=circ_std_deg,
        rms_angle_deviation_from_mean_deg=rms_delta,
        min_angle_deviation_deg=min_delta,
        max_angle_deviation_deg=max_delta,
        peak_to_peak_deviation_deg=float(max_delta - min_delta) if np.isfinite(min_delta) else np.nan,
        circular_mean_psi_deg_PoverI_weighted=mean_psi_Pw,
    )

    print("\n" + "=" * 70)
    print("APPARENT POLARIZATION-ANGLE MAP STATISTICS (inside beam)")
    print("  psi(u,v) = 0.5 * atan2(UT, QT)  [deg]")
    print("  Circular statistics used (double-angle trick) for 180 deg ambiguity.")
    print(f"  Mask threshold: TT > {maps_dict['tt_threshold_frac']:.0e} * TT_peak")
    print(f"  Valid pixels: {results['n_valid_pixels']}")
    print("=" * 70)
    print(f"  Circular mean psi (TT-weighted)   = {mean_psi:.4f} deg")
    print(f"  Circular std psi  (TT-weighted)   = {circ_std_deg:.4f} deg")
    print(f"  RMS deviation from mean (TT-wtd)  = {rms_delta:.6f} deg")
    print(f"  Min angle deviation                = {min_delta:.6f} deg")
    print(f"  Max angle deviation                = {max_delta:.6f} deg")
    print(f"  Peak-to-peak deviation             = {max_delta - min_delta:.6f} deg")
    print(f"  Circular mean psi (P/I-weighted)  = {mean_psi_Pw:.4f} deg")
    print("=" * 70)

    return results


# ---------------------------------------------------------------------------
# Step 8 -- Ideal partial-polarizer model and residuals
# ---------------------------------------------------------------------------

def build_ideal_model_and_residuals(mueller_nogrid, mueller_grid, mm_grid, u, v):
    """Construct an ideal partial-polarizer prediction and compute residuals.

    Model: M_ideal(u,v) = TT_nogrid(u,v) * M_int_grid
           (a spatially uniform polarizer scaled by the no-grid beam shape)

    Since the no-grid case is approximately identity times the beam pattern,
    TT_nogrid(u,v) carries the beam shape.  The integrated M_int_grid has
    M[0,0] = 1 by construction, so the TT of the model also integrates to 1.

    Residual: R(u,v) = M_grid(u,v) - M_ideal(u,v)

    Convention note: we multiply a constant 4x4 matrix by a scalar field.
    This models a spatially uniform polarization response (ideal case) and
    leaves any spatially varying polarization effects in the residual.
    """
    TT_nogrid = mueller_nogrid[:, :, 0, 0]  # shape (nu, nv)

    # Broadcast: TT_nogrid[:, :, None, None] * mm_grid[None, None, :, :]
    M_ideal = TT_nogrid[:, :, None, None] * mm_grid[None, None, :, :]

    residual = mueller_grid - M_ideal

    print("\n[Ideal model] Built TT_nogrid * M_int_grid as ideal partial-polarizer prediction.")
    print(f"  Residual max abs value: {np.abs(residual).max():.4e}")
    print(f"  Residual RMS:           {np.sqrt(np.mean(residual**2)):.4e}")

    return M_ideal, residual


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def _colorbar(ax, im, label=None):
    cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    if label:
        cb.set_label(label, fontsize=7)


def plot_matrix_heatmap(mm, title, ax=None, vmax=None):
    """Plot a 4x4 Mueller matrix as a colour-coded heatmap."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    if vmax is None:
        vmax = max(np.abs(mm).max(), 1e-12)

    im = ax.imshow(mm, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_xticks(range(4))
    ax.set_xticklabels(STOKES, fontsize=9)
    ax.set_yticks(range(4))
    ax.set_yticklabels(STOKES, fontsize=9)
    ax.set_xlabel("Input Stokes", fontsize=8)
    ax.set_ylabel("Output Stokes", fontsize=8)
    ax.set_title(title, fontsize=9)
    _colorbar(ax, im)

    # Annotate values
    for r in range(4):
        for c in range(4):
            ax.text(c, r, f"{mm[r, c]:.3f}", ha="center", va="center",
                    fontsize=7, color="black" if abs(mm[r, c]) < 0.5 * vmax else "white")
    return fig


def plot_three_heatmaps(mm_nogrid, mm_grid, outdir, pdf=None):
    """Integrated Mueller matrix heatmaps: nogrid, grid, difference."""
    diff = mm_grid - mm_nogrid
    vmax = max(np.abs(mm_grid).max(), np.abs(diff).max(), 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for mm, ax, title in [
        (mm_nogrid, axes[0], "No-grid"),
        (mm_grid,   axes[1], "Wire-grid"),
        (diff,      axes[2], "Difference (grid - nogrid)"),
    ]:
        lv = max(np.abs(mm).max(), 1e-12)
        im = ax.imshow(mm, cmap="RdBu_r", vmin=-lv, vmax=lv, aspect="equal")
        ax.set_xticks(range(4))
        ax.set_xticklabels(STOKES, fontsize=9)
        ax.set_yticks(range(4))
        ax.set_yticklabels(STOKES, fontsize=9)
        ax.set_xlabel("Input Stokes", fontsize=8)
        ax.set_ylabel("Output Stokes", fontsize=8)
        ax.set_title(title, fontsize=10)
        _colorbar(ax, im)
        for r in range(4):
            for c in range(4):
                ax.text(c, r, f"{mm[r,c]:.3f}", ha="center", va="center",
                        fontsize=7, color="black" if abs(mm[r,c]) < 0.5*lv else "white")

    fig.suptitle("Integrated Mueller Matrices (normalised, M_TT = 1)", fontsize=12)
    fig.savefig(outdir / "matrix_heatmaps.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {outdir / 'matrix_heatmaps.png'}")
    if pdf:
        pdf.savefig(fig)
    plt.close(fig)


def plot_first_column_maps(maps_dict, u, v, outdir, pdf=None):
    """TT, QT, UT, VT spatial maps (raw normalised values)."""
    ext = _extent_arcmin(u, v)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)

    for ax, key, label in zip(axes, ["TT", "QT", "UT", "VT"],
                               ["$M_{TT}$", "$M_{QT}$", "$M_{UT}$", "$M_{VT}$"]):
        m = maps_dict[key]
        if key == "TT":
            im = ax.imshow(m.T, origin="lower", extent=ext,
                           cmap="inferno", aspect="equal",
                           vmin=0, vmax=m.max() if m.max() > 0 else 1)
        else:
            lv = max(np.abs(m).max(), 1e-12)
            im = ax.imshow(m.T, origin="lower", extent=ext,
                           cmap="RdBu_r", aspect="equal",
                           norm=CenteredNorm(halfrange=lv))
        ax.set_title(f"Wire-grid {label}", fontsize=10)
        ax.set_xlabel("u (arcmin)", fontsize=8)
        ax.set_ylabel("v (arcmin)", fontsize=8)
        ax.tick_params(labelsize=7)
        _colorbar(ax, im)

    fig.suptitle("Wire-grid First Column: response to unpolarized input S_in=(1,0,0,0)",
                 fontsize=11)
    fig.savefig(outdir / "first_column_maps.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {outdir / 'first_column_maps.png'}")
    if pdf:
        pdf.savefig(fig)
    plt.close(fig)


def plot_normalized_first_column(maps_dict, u, v, outdir, pdf=None):
    """QT/TT, UT/TT, P/I, psi(u,v) maps with masked low-signal pixels."""
    ext = _extent_arcmin(u, v)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)

    panels = [
        ("QT_over_TT", "$Q_T / T_T$", "RdBu_r", None),
        ("UT_over_TT", "$U_T / T_T$", "RdBu_r", None),
        ("P_over_I",   "$P/I$",       "plasma",  (0, None)),
        ("psi_deg",    "$\\psi$ (deg)", "hsv",   (-90, 90)),
    ]

    for ax, (key, label, cmap, clim) in zip(axes, panels):
        m = maps_dict[key]
        vmin = clim[0] if clim else None
        vmax = clim[1] if (clim and clim[1] is not None) else None

        if key == "psi_deg":
            # Symmetric around zero; force symmetric colour limits
            finite = m[np.isfinite(m)]
            lv = max(np.abs(finite).max(), 1e-12) if finite.size > 0 else 1
            im = ax.imshow(m.T, origin="lower", extent=ext,
                           cmap=cmap, aspect="equal", vmin=-lv, vmax=lv)
        elif key == "P_over_I":
            finite = m[np.isfinite(m)]
            vmax_auto = finite.max() if finite.size > 0 else 1
            im = ax.imshow(m.T, origin="lower", extent=ext,
                           cmap=cmap, aspect="equal", vmin=0, vmax=vmax_auto)
        else:
            finite = m[np.isfinite(m)]
            lv = max(np.abs(finite).max(), 1e-12) if finite.size > 0 else 1
            im = ax.imshow(m.T, origin="lower", extent=ext,
                           cmap=cmap, aspect="equal", vmin=-lv, vmax=lv)

        ax.set_title(label, fontsize=10)
        ax.set_xlabel("u (arcmin)", fontsize=8)
        ax.set_ylabel("v (arcmin)", fontsize=8)
        ax.tick_params(labelsize=7)
        _colorbar(ax, im)

    thresh = maps_dict["tt_threshold_frac"]
    fig.suptitle(
        f"Wire-grid Normalized First-Column Maps  "
        f"(masked: $T_T < {thresh:.0e}\\,T_{{T,\\mathrm{{peak}}}}$)",
        fontsize=11,
    )
    fig.savefig(outdir / "normalized_first_column.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {outdir / 'normalized_first_column.png'}")
    if pdf:
        pdf.savefig(fig)
    plt.close(fig)


def plot_residual_maps(residual, u, v, outdir, pdf=None):
    """4x4 grid of residual Mueller maps after subtracting the ideal model."""
    ext = _extent_arcmin(u, v)
    fig, axes = plt.subplots(4, 4, figsize=(16, 16), constrained_layout=True)

    for r in range(4):
        for c in range(4):
            ax = axes[r, c]
            m = residual[:, :, r, c]
            lv = max(np.abs(m).max(), 1e-12)
            im = ax.imshow(m.T, origin="lower", extent=ext,
                           cmap="RdBu_r", aspect="equal",
                           norm=CenteredNorm(halfrange=lv))
            ax.set_title(f"R_{STOKES[r]}{STOKES[c]}", fontsize=8)
            ax.set_xlabel("u (arcmin)", fontsize=6)
            ax.set_ylabel("v (arcmin)", fontsize=6)
            ax.tick_params(labelsize=5)
            _colorbar(ax, im)

    fig.suptitle(
        "Residual Mueller Maps\n"
        "R(u,v) = M_grid(u,v)  -  TT_nogrid(u,v) * M_int_grid\n"
        "(deviations from an ideal, spatially-uniform partial-polarizer model)",
        fontsize=11,
    )
    fig.savefig(outdir / "residual_maps.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {outdir / 'residual_maps.png'}")
    if pdf:
        pdf.savefig(fig)
    plt.close(fig)


def plot_mueller_maps_4x4(mueller, u, v, title, filename, outdir, pdf=None):
    """Standard 4x4 Mueller element maps (reuses the same style as run_single_beam)."""
    ext = _extent_arcmin(u, v)
    fig, axes = plt.subplots(4, 4, figsize=(16, 16), constrained_layout=True)

    for r in range(4):
        for c in range(4):
            ax = axes[r, c]
            m = mueller[:, :, r, c]
            if r == c:
                im = ax.imshow(m.T, origin="lower", extent=ext,
                               cmap="inferno", aspect="equal",
                               vmin=0, vmax=m.max() if m.max() > 0 else 1)
            else:
                lv = max(np.abs(m).max(), 1e-12)
                im = ax.imshow(m.T, origin="lower", extent=ext,
                               cmap="RdBu_r", aspect="equal",
                               norm=CenteredNorm(halfrange=lv))
            ax.set_title(f"M_{STOKES[r]}{STOKES[c]}", fontsize=8)
            ax.set_xlabel("u (arcmin)", fontsize=6)
            ax.set_ylabel("v (arcmin)", fontsize=6)
            ax.tick_params(labelsize=5)
            _colorbar(ax, im)

    fig.suptitle(title, fontsize=12)
    fig.savefig(outdir / filename, dpi=150, bbox_inches="tight")
    print(f"  Saved: {outdir / filename}")
    if pdf:
        pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_summary(pp_results, unp_results, angle_stats, outdir):
    """Write summary.json, summary.csv, summary.txt."""
    summary = {}
    summary["partial_polarizer"] = pp_results
    summary["unpolarized_input_integrated"] = unp_results
    summary["angle_map_statistics"] = angle_stats

    json_path = outdir / "summary.json"
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    print(f"  Saved: {json_path}")

    # Flat CSV
    csv_path = outdir / "summary.csv"
    rows = []
    for section, vals in summary.items():
        for k, v in vals.items():
            rows.append({"section": section, "quantity": k, "value": v})
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["section", "quantity", "value"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {csv_path}")

    # Human-readable text with interpretation notes
    txt_path = outdir / "summary.txt"
    with open(txt_path, "w") as fh:
        fh.write("Wire-grid Mueller Non-Ideal Analysis -- Summary\n")
        fh.write("=" * 60 + "\n\n")

        fh.write("CONVENTION\n")
        fh.write("  M[r,c] maps Stokes-c input to Stokes-r output.\n")
        fh.write("  Maps normalised: integral of M_TT = 1.\n")
        fh.write("  M_int[0,0] = 1 by construction.\n\n")

        fh.write("PARTIAL-POLARIZER PARAMETERS\n")
        fh.write(f"  p_TQ = M_TQ / M_TT = {pp_results['p_TQ']:+.6f}\n")
        fh.write(f"  p_QT = M_QT / M_TT = {pp_results['p_QT']:+.6f}\n")
        fh.write(f"  |p|  = {pp_results['abs_p']:.6f}\n")
        fh.write(f"  T_blocked / T_passed = {pp_results['T_blocked_over_T_passed']:.6f} ({pp_results['T_blocked_percent']:.2f}%)\n")
        fh.write(f"  UV retardation angle = {pp_results['retardation_angle_deg']:.2f} deg\n\n")

        fh.write("UNPOLARIZED INPUT (integrated)\n")
        fh.write(f"  QT/TT (I->Q leakage) = {unp_results['QT_over_TT']:+.6f} ({unp_results['QT_over_TT']*100:.2f}%)\n")
        fh.write(f"  UT/TT (I->U leakage) = {unp_results['UT_over_TT']:+.6f} ({unp_results['UT_over_TT']*100:.4f}%)\n")
        fh.write(f"  P/I                  = {unp_results['P_over_I']:.6f} ({unp_results['P_over_I']*100:.2f}%)\n")
        fh.write(f"  Apparent pol. angle  = {unp_results['psi_eff_deg']:.4f} deg\n\n")

        fh.write("ANGLE MAP STATISTICS (TT-weighted, circular mean, inside main beam)\n")
        fh.write(f"  Circular mean psi    = {angle_stats['circular_mean_psi_deg']:.4f} deg\n")
        fh.write(f"  Circular std psi     = {angle_stats['circular_std_psi_deg']:.4f} deg\n")
        fh.write(f"  RMS deviation        = {angle_stats['rms_angle_deviation_from_mean_deg']:.6f} deg\n")
        fh.write(f"  Min / max deviation  = {angle_stats['min_angle_deviation_deg']:.6f} / {angle_stats['max_angle_deviation_deg']:.6f} deg\n")
        fh.write(f"  Peak-to-peak dev.    = {angle_stats['peak_to_peak_deviation_deg']:.6f} deg\n\n")

        fh.write("INTERPRETATION NOTES\n")
        fh.write("  - No-grid case: M_int ~ identity. The feed is a near-ideal\n")
        fh.write("    polarimeter with ~1e-6 leakage.\n")
        fh.write("  - Wire-grid case: large I<->Q coupling (M_TQ ~ M_QT ~ -0.83).\n")
        fh.write("    This is the expected signature of a partial linear polarizer\n")
        fh.write("    and is the DESIRED behaviour for a grid-based calibration source.\n")
        fh.write("  - The I->Q term is large and expected. The KEY non-ideal\n")
        fh.write("    diagnostics are:\n")
        fh.write("      * I->U (UT/TT): should be ~0 for a perfectly-aligned grid.\n")
        fh.write("      * Spatially-varying polarization angle psi(u,v): RMS > ~0.1 deg\n")
        fh.write("        indicates beam-dependent polarization rotation.\n")
        fh.write("      * Residuals after ideal-model subtraction: reveal edge effects,\n")
        fh.write("        diffraction, and near-field wire-interaction non-uniformities.\n")
        fh.write("  - UV retardation angle: differential phase delay between U and V\n")
        fh.write("    channels introduced by the grid structure.\n")
    print(f"  Saved: {txt_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--nogrid-pol1", required=True, metavar="PATH",
        help="No-grid .grd file for x-pol (pol1) drive.",
    )
    parser.add_argument(
        "--nogrid-pol2", required=True, metavar="PATH",
        help="No-grid .grd file for y-pol (pol2) drive.",
    )
    parser.add_argument(
        "--grid-pol1", required=True, metavar="PATH",
        help="Wire-grid .grd file for x-pol (pol1) drive.",
    )
    parser.add_argument(
        "--grid-pol2", required=True, metavar="PATH",
        help="Wire-grid .grd file for y-pol (pol2) drive.",
    )
    parser.add_argument(
        "--outdir", default="outputs/wiregrid_mueller_nonideal", metavar="DIR",
        help="Output directory (created if absent). Default: outputs/wiregrid_mueller_nonideal",
    )
    parser.add_argument(
        "--grd-units", default="auto", choices=["auto", "rad", "deg"],
        help="Coordinate units in .grd files (default: auto-detect from ICOORD header).",
    )
    parser.add_argument(
        "--tt-threshold", type=float, default=1e-3, metavar="FRAC",
        help="Beam-map mask threshold: only evaluate angle/ratio maps where "
             "TT > FRAC * TT_peak. Default: 1e-3.",
    )
    parser.add_argument(
        "--no-pdf", action="store_true",
        help="Skip writing combined summary.pdf (individual PNGs are always written).",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {outdir.resolve()}")

    # ------------------------------------------------------------------
    # Steps 1 & 2 -- Load and build Mueller maps
    # ------------------------------------------------------------------
    u_ng, v_ng, mueller_nogrid, mm_nogrid = load_case(
        args.nogrid_pol1, args.nogrid_pol2, "no-grid", args.grd_units
    )
    u_gr, v_gr, mueller_grid, mm_grid = load_case(
        args.grid_pol1, args.grid_pol2, "wire-grid", args.grd_units
    )

    # Verify grids are compatible
    if not (np.allclose(u_ng, u_gr) and np.allclose(v_ng, v_gr)):
        print("\nWARNING: no-grid and wire-grid coordinate axes differ -- "
              "residual maps may be unreliable.")

    u, v = u_ng, v_ng  # use nogrid axes as reference

    # Save Mueller maps and integrated matrices
    print("\n[Saving] Mueller maps and integrated matrices...")
    np.save(outdir / "mueller_maps_nogrid.npy", mueller_nogrid)
    np.save(outdir / "mueller_maps_grid.npy", mueller_grid)
    np.save(outdir / "mueller_matrix_nogrid.npy", mm_nogrid)
    np.save(outdir / "mueller_matrix_grid.npy", mm_grid)
    np.save(outdir / "mueller_matrix_difference.npy", mm_grid - mm_nogrid)
    print(f"  Saved: mueller_maps_nogrid.npy, mueller_maps_grid.npy")
    print(f"  Saved: mueller_matrix_nogrid.npy, mueller_matrix_grid.npy, mueller_matrix_difference.npy")

    # ------------------------------------------------------------------
    # Step 3 -- Comparison printout
    # ------------------------------------------------------------------
    print_comparison(mm_nogrid, mm_grid)

    # ------------------------------------------------------------------
    # Step 4 -- Partial-polarizer analysis
    # ------------------------------------------------------------------
    pp_results = partial_polarizer_analysis(mm_grid)

    # ------------------------------------------------------------------
    # Step 5 -- Unpolarized-input integrated diagnostics
    # ------------------------------------------------------------------
    unp_results = unpolarized_input_analysis(mm_grid)

    # ------------------------------------------------------------------
    # Steps 6 & 7 -- Beam-map first-column quantities and angle statistics
    # ------------------------------------------------------------------
    print(f"\n[Maps] Computing first-column beam maps (threshold = {args.tt_threshold:.0e})...")
    maps = first_column_maps(mueller_grid, u, v, tt_threshold_frac=args.tt_threshold)
    angle_stats = beam_weighted_angle_stats(maps)

    # ------------------------------------------------------------------
    # Step 8 -- Ideal model and residuals
    # ------------------------------------------------------------------
    print("\n[Residuals] Building ideal partial-polarizer model and residual maps...")
    M_ideal, residual = build_ideal_model_and_residuals(
        mueller_nogrid, mueller_grid, mm_grid, u, v
    )
    np.save(outdir / "mueller_maps_ideal_model.npy", M_ideal)
    np.save(outdir / "mueller_maps_residual.npy", residual)
    print(f"  Saved: mueller_maps_ideal_model.npy, mueller_maps_residual.npy")

    # ------------------------------------------------------------------
    # Step 9 -- Plots
    # ------------------------------------------------------------------
    print("\n[Plots] Generating figures...")

    pdf_path = outdir / "summary.pdf"
    pdf_ctx = PdfPages(pdf_path) if not args.no_pdf else None

    try:
        plot_three_heatmaps(mm_nogrid, mm_grid, outdir, pdf=pdf_ctx)
        plot_mueller_maps_4x4(
            mueller_nogrid, u, v,
            "No-grid Mueller Maps (normalised)",
            "mueller_maps_nogrid.png", outdir, pdf=pdf_ctx,
        )
        plot_mueller_maps_4x4(
            mueller_grid, u, v,
            "Wire-grid Mueller Maps (normalised)",
            "mueller_maps_grid.png", outdir, pdf=pdf_ctx,
        )
        plot_first_column_maps(maps, u, v, outdir, pdf=pdf_ctx)
        plot_normalized_first_column(maps, u, v, outdir, pdf=pdf_ctx)
        plot_residual_maps(residual, u, v, outdir, pdf=pdf_ctx)
    finally:
        if pdf_ctx is not None:
            pdf_ctx.close()
            print(f"  Saved: {pdf_path}")

    # ------------------------------------------------------------------
    # Step 10 -- Machine-readable summary
    # ------------------------------------------------------------------
    print("\n[Summary] Writing text/JSON/CSV outputs...")
    save_summary(pp_results, unp_results, angle_stats, outdir)

    # ------------------------------------------------------------------
    # Final listing
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ALL OUTPUTS")
    print("=" * 70)
    for f in sorted(outdir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:45s}  {size_kb:8.1f} kB")
    print("=" * 70)
    print("\nDone.")


if __name__ == "__main__":
    main()
