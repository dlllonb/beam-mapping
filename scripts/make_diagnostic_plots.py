"""
make_diagnostic_plots.py  --  Publication-quality diagnostic figures for beam-map analysis.

Reproduces the most scientifically important plots from the original IDL GRASP
beam-analysis code, focused on:

  Figure 1  Focal-plane polarization-angle rotation map (IDL CW/CCW map)
  Figure 2  Distribution of induced polarization-angle rotation (IDL histogram panel 3)
  Figure 3  Integrated Mueller leakage maps: QT, UT, QU, UQ (IDL Mueller focal-plane maps)
  Figure 4  Differential beam mismatch between polarization channels (IDL beam mismatch)
  Figure 5+ Representative Mueller beam maps for selected feeds (IDL per-feed beam maps)

The science question: can diffraction / off-axis optical effects from the aperture
stop or surrounding optics induce polarization-angle errors, Stokes Q/U mixing,
sidelobes, or false B-mode contamination?

Usage
-----
From Python:

    import numpy as np
    from scripts.make_diagnostic_plots import plot_diagnostic_figures

    results = np.load("array_data.npz", allow_pickle=True)
    plot_diagnostic_figures(results, output_pdf="diagnostic_figures.pdf")

    # Or save individual PNGs:
    plot_diagnostic_figures(results, output_dir="diagnostic_figures/")

From the command line:

    python scripts/make_diagnostic_plots.py array_data.npz --output-pdf diagnostic_figures.pdf
    python scripts/make_diagnostic_plots.py array_data.npz --output-dir diagnostic_figures/

Input
-----
The .npz file produced by array_map.py, with keys:
    mueller_matrices  (18, n_feeds)  rows 0-15: integrated Mueller elements,
                                     row 16: phi_x_deg, row 17: phi_y_deg
    mueller_maps      (nu, nv, 16, n_feeds)  pixel-level Mueller maps (optional,
                                              needed for Figure 5 only)
    beam_data         (17, n_feeds)  beam characterisation metrics
    u, v              1-D coordinate arrays in radians
    names             feed identifier strings
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import CenteredNorm, Normalize
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import EFL_MM, DEG2RAD, RAD2ARCMIN, HORN_LAYOUT_FILE
from core.read_horn_layout import load_layout

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default output directory — resolved relative to this script file so it
# always lands in beam-mapping/outputs/ regardless of working directory.
_OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

STOKES = ["T", "Q", "U", "V"]

# Flat Mueller index: element XY (output X, input Y) -> row*4+col
# Order: TT=0, TQ=1, TU=2, TV=3, QT=4, QQ=5, QU=6, QV=7,
#        UT=8, UQ=9, UU=10, UV=11, VT=12, VQ=13, VU=14, VV=15
_IDX = {
    f"{r}{c}": ri * 4 + ci
    for ri, r in enumerate(STOKES)
    for ci, c in enumerate(STOKES)
}

# Publication font sizes
_FS_TITLE  = 11
_FS_LABEL  = 10
_FS_TICK   =  8
_FS_CBAR   =  9
_FS_ANNOT  =  8
_FS_LEGEND =  8


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def _load_results(results):
    """Accept a path/str, numpy NpzFile, or plain dict."""
    if isinstance(results, (str, Path)):
        return np.load(results, allow_pickle=True)
    return results


def _names(results):
    """Return feed names as a list of str (handles bytes from numpy)."""
    raw = np.asarray(results["names"])
    if raw.dtype.kind in ("S", "U", "O"):
        return [n.decode() if isinstance(n, bytes) else str(n) for n in raw]
    return [str(n) for n in raw]


# ---------------------------------------------------------------------------
# Focal-plane coordinates
# ---------------------------------------------------------------------------

def _focal_plane_mm(results, horn_layout_path=None):
    """Return (x_mm, y_mm, names_list) for all feeds in *results*.

    Matching strategy (in order of preference):
      1. Load horn-layout file and match by normalised name (hyphens stripped),
         which handles filenames like '331-F1' vs layout entries '331F1'.
      2. Fall back to sky-offset back-conversion: x_mm = phi_x_deg * DEG2RAD * EFL_MM.

    Warns clearly if the resulting coordinates are all zero (which means the
    horn-layout match AND the sky-offset fallback both failed).
    """
    names = _names(results)
    phi_x = np.asarray(results["mueller_matrices"][16], dtype=float)
    phi_y = np.asarray(results["mueller_matrices"][17], dtype=float)

    def _norm(n):
        return n.replace("-", "").replace(" ", "")

    hl_path = horn_layout_path or HORN_LAYOUT_FILE
    try:
        horn_df = load_layout(hl_path, efl_mm=EFL_MM)
        # Build lookup with normalised keys so '331-F1' matches '331F1'
        name_x = {_norm(n): x for n, x in zip(horn_df["name"], horn_df["x_mm"])}
        name_y = {_norm(n): y for n, y in zip(horn_df["name"], horn_df["y_mm"])}
        x_mm = np.array([name_x.get(_norm(n), np.nan) for n in names])
        y_mm = np.array([name_y.get(_norm(n), np.nan) for n in names])
        n_matched = int(np.isfinite(x_mm).sum())
        print(f"  [make_diagnostic_plots] Horn-layout: matched {n_matched}/{len(names)} feeds "
              f"from {hl_path.name}")
        if n_matched < max(1, len(names) // 2):
            raise ValueError(f"Only {n_matched} of {len(names)} names matched; "
                             "falling back to sky-offset conversion.")
    except Exception as exc:
        print(f"  [make_diagnostic_plots] Horn-layout fallback: {exc}")
        x_mm = phi_x * DEG2RAD * EFL_MM
        y_mm = phi_y * DEG2RAD * EFL_MM

    # Sanity check: warn if everything is at the origin
    if np.nanmax(np.abs(x_mm)) < 1.0 and np.nanmax(np.abs(y_mm)) < 1.0:
        print("  [make_diagnostic_plots] WARNING: all feed x/y coordinates are ~0 mm. "
              "The focal-plane map will appear blank.\n"
              "  This usually means sky offsets in the .npz are all zero because\n"
              "  array_map.py could not match feed names to the horn layout.\n"
              "  Check that feed filenames match the horn_layout_331.dat name format.")

    return x_mm, y_mm, names


# ---------------------------------------------------------------------------
# Polarization-rotation helper
# ---------------------------------------------------------------------------

def compute_pol_rotation(results):
    """Per-feed induced polarization-angle rotation [degrees].

    Matches the IDL formula from plot_array_map_v2026.03.17.pro (lines ~573):

        S2 = MM ## [1, 1, 0, 0]          ; apply (T+Q) input Stokes vector
        rot_angle = (180/pi) * atan(S2[U] / S2[Q]) / 2

    which expands to:

        rot_angle = degrees( arctan2(UT + UQ,  QT + QQ) ) / 2

    where UT, UQ, QT, QQ are integrated Mueller elements (output_row, input_col).

    Sign convention (from IDL source):
        positive rot_angle → CW rotation → RED in the focal-plane map
        negative rot_angle → CCW rotation → BLUE in the focal-plane map

    This is the opposite of the standard mathematical sign convention, so the
    focal-plane plot uses the 'RdBu' colormap (negative → red, positive → blue
    is the standard matplotlib direction; we want positive → red here, so the
    sign may need to be validated).  A sign-flip line is left commented below.
    Validate against the IDL PDF before publishing.
    """
    mm = results["mueller_matrices"]
    QT = np.asarray(mm[_IDX["QT"]], dtype=float)
    QQ = np.asarray(mm[_IDX["QQ"]], dtype=float)
    UT = np.asarray(mm[_IDX["UT"]], dtype=float)
    UQ = np.asarray(mm[_IDX["UQ"]], dtype=float)

    # IDL: rot_angle = (180/pi) * atan( (UT+UQ) / (QT+QQ) ) / 2
    psi_rot_deg = np.degrees(np.arctan2(UT + UQ, QT + QQ)) / 2.0

    # Uncomment to flip sign if upper-half feeds appear blue instead of red:
    # psi_rot_deg = -psi_rot_deg

    return psi_rot_deg


# ---------------------------------------------------------------------------
# Shared focal-plane plot helpers
# ---------------------------------------------------------------------------

def _style_focal_ax(ax, title):
    """Apply consistent focal-plane axis style."""
    ax.set_xlabel("x [mm]", fontsize=_FS_LABEL)
    ax.set_ylabel("y [mm]", fontsize=_FS_LABEL)
    ax.tick_params(labelsize=_FS_TICK)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=_FS_TITLE, pad=6)
    ax.axhline(0, color="0.65", lw=0.7, ls="--", zorder=0)
    ax.axvline(0, color="0.65", lw=0.7, ls="--", zorder=0)


def _draw_focal_boundary(ax, x_mm, y_mm):
    """Dotted circle slightly beyond the outermost feed."""
    r_max = np.nanmax(np.sqrt(x_mm ** 2 + y_mm ** 2))
    if not (np.isfinite(r_max) and r_max > 0):
        return
    circle = plt.Circle(
        (0, 0), r_max * 1.07,
        fill=False, ls=":", color="0.45", lw=1.0, zorder=1
    )
    ax.add_patch(circle)


def _focal_scatter(ax, x_mm, y_mm, values, title, cbar_label,
                   cmap="RdBu_r", symmetric=True, s=55, vmin=None, vmax=None):
    """Scatter on focal-plane coordinates with a labelled colorbar."""
    valid = np.isfinite(values) & np.isfinite(x_mm) & np.isfinite(y_mm)
    if not valid.any():
        ax.set_title(title + " (no data)", fontsize=_FS_TITLE)
        return None

    v = values[valid]
    if symmetric:
        half = max(float(np.abs(v).max()), 1e-12)
        norm = CenteredNorm(halfrange=half)
    else:
        lo = vmin if vmin is not None else float(v.min())
        hi = vmax if vmax is not None else float(v.max())
        norm = Normalize(vmin=lo, vmax=hi)

    sc = ax.scatter(
        x_mm[valid], y_mm[valid], c=v,
        cmap=cmap, norm=norm, s=s,
        edgecolors="none", linewidths=0, zorder=3,
    )
    cb = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label(cbar_label, fontsize=_FS_CBAR)
    cb.ax.tick_params(labelsize=_FS_TICK)
    _style_focal_ax(ax, title)
    return sc


# ---------------------------------------------------------------------------
# Figure 1: Focal-plane polarization-angle rotation map
# IDL reference: CW/CCW focal-plane polarization rotation map
# ---------------------------------------------------------------------------

def fig_pol_rotation_map(results, x_mm, y_mm):
    """Focal-plane map with circle size ∝ |rotation| and color = sign.

    IDL convention: positive rotation → CW → RED (upper half of focal plane).
                    negative rotation → CCW → BLUE (lower half).
    RdBu_r maps positive values to red, which matches this convention.
    If the upper half appears blue instead of red, uncomment the sign-flip
    in compute_pol_rotation().
    """
    psi_deg = compute_pol_rotation(results)
    valid = np.isfinite(psi_deg) & np.isfinite(x_mm) & np.isfinite(y_mm)

    fig, ax = plt.subplots(figsize=(7, 7))

    if valid.any():
        pv = psi_deg[valid]
        half = max(float(np.abs(pv).max()), 1e-12)

        # Marker size proportional to |rotation|; minimum keeps all feeds visible
        S_MIN, S_MAX = 15.0, 250.0
        sizes = S_MIN + (np.abs(pv) / half) * (S_MAX - S_MIN)

        norm = CenteredNorm(halfrange=half)
        sc = ax.scatter(
            x_mm[valid], y_mm[valid], c=pv,
            s=sizes, cmap="RdBu_r", norm=norm,
            edgecolors="0.35", linewidths=0.3, zorder=3,
        )
        cb = plt.colorbar(sc, ax=ax, pad=0.03, fraction=0.046)
        cb.set_label("polarization rotation [deg]", fontsize=_FS_CBAR)
        cb.ax.tick_params(labelsize=_FS_TICK)

        # Legend entries  (IDL: positive = CW = red; RdBu_r: positive → red)
        cw_patch  = mpatches.Patch(color="#C03020", label="CW (positive)")
        ccw_patch = mpatches.Patch(color="#2060C0", label="CCW (negative)")
        # Scale marker at 0.2 deg
        scale_val = 0.2
        scale_s   = S_MIN + (scale_val / half) * (S_MAX - S_MIN)
        scale_pt  = ax.scatter([], [], s=scale_s, c="0.55",
                               edgecolors="0.35", linewidths=0.3,
                               label=f"scale: {scale_val:.1f} deg")
        ax.legend(
            handles=[cw_patch, ccw_patch, scale_pt],
            fontsize=_FS_LEGEND, loc="upper right", framealpha=0.85,
        )

    _draw_focal_boundary(ax, x_mm, y_mm)
    _style_focal_ax(ax, "Induced polarization-angle rotation across focal plane")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2: Polarization-angle rotation histogram
# IDL reference: third panel of the IDL histogram figure
# ---------------------------------------------------------------------------

def fig_pol_rotation_hist(results):
    """Step histogram of induced polarization-angle rotation across all feeds."""
    psi_deg = compute_pol_rotation(results)
    valid = psi_deg[np.isfinite(psi_deg)]

    fig, ax = plt.subplots(figsize=(7, 5))

    if len(valid) == 0:
        ax.set_title("No valid data", fontsize=_FS_TITLE)
        fig.tight_layout()
        return fig

    # IDL uses 31 bins over [-1.5, 1.5]; mirror that range but allow data outside
    _ROT_REQ = 0.2   # IDL rot_requirement [deg] — vertical reference lines
    xmax = max(1.5, float(np.nanpercentile(np.abs(valid), 99)) * 1.1)
    n_bins = 31
    ax.hist(valid, bins=n_bins, range=(-xmax, xmax),
            histtype="step", color="black", lw=1.5)

    mean_r   = float(np.mean(valid))
    median_r = float(np.median(valid))
    rms_r    = float(np.sqrt(np.mean(valid ** 2)))
    max_abs  = float(np.max(np.abs(valid)))
    p95_abs  = float(np.percentile(np.abs(valid), 95))

    # Reference lines matching IDL: dashed red at ±rot_requirement (0.2 deg)
    ax.axvline(0,           color="black",   lw=0.8, ls="-",  alpha=0.4)
    ax.axvline(-_ROT_REQ,   color="crimson", lw=1.4, ls=(0, (6, 3, 1, 3)),
               label=f"±{_ROT_REQ:.1f} deg requirement")
    ax.axvline(+_ROT_REQ,   color="crimson", lw=1.4, ls=(0, (6, 3, 1, 3)))

    stats_text = (
        f"RMS      = {rms_r:.3f} deg\n"
        f"max |ψ|  = {max_abs:.3f} deg\n"
        f"|ψ| 95th = {p95_abs:.3f} deg"
    )
    ax.text(
        0.97, 0.97, stats_text, transform=ax.transAxes,
        ha="right", va="top", fontsize=_FS_ANNOT,
        family="monospace",
        bbox=dict(fc="white", ec="0.65", alpha=0.9, boxstyle="round,pad=0.3"),
    )

    ax.set_xlim(-xmax, xmax)
    ax.set_xlabel("polarization rotation [deg]", fontsize=_FS_LABEL)
    ax.set_ylabel("number of feeds",             fontsize=_FS_LABEL)
    ax.set_title("Distribution of induced polarization-angle rotation",
                 fontsize=_FS_TITLE)
    ax.tick_params(labelsize=_FS_TICK, direction="in", top=True, right=True)
    ax.minorticks_on()
    ax.legend(fontsize=_FS_LEGEND)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3: Integrated Mueller leakage maps: QT, UT, QU, UQ
# IDL reference: integrated Mueller focal-plane scatter maps
# ---------------------------------------------------------------------------

def fig_mueller_leakage_maps(results, x_mm, y_mm):
    """2x2 focal-plane panels for QT, UT, QU, UQ.

    QT, UT = temperature-to-polarization leakage (dangerous because T >> B).
    QU, UQ = Q/U mixing, directly related to E/B leakage and pol-angle rotation.
    """
    mm = results["mueller_matrices"]
    elements = [
        ("QT", np.asarray(mm[_IDX["QT"]], dtype=float)),
        ("UT", np.asarray(mm[_IDX["UT"]], dtype=float)),
        ("QU", np.asarray(mm[_IDX["QU"]], dtype=float)),
        ("UQ", np.asarray(mm[_IDX["UQ"]], dtype=float)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    fig.suptitle(
        "Integrated Mueller leakage terms relevant to polarization systematics",
        fontsize=_FS_TITLE + 1,
    )

    for ax, (name, vals) in zip(axes.ravel(), elements):
        _focal_scatter(ax, x_mm, y_mm, vals,
                       title=name,
                       cbar_label=f"integrated Mueller response  [{name}]",
                       cmap="RdBu_r", symmetric=True)
        _draw_focal_boundary(ax, x_mm, y_mm)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4: Differential beam mismatch
# IDL reference: beam mismatch diagnostics
# ---------------------------------------------------------------------------

def fig_differential_beam_mismatch(results, x_mm, y_mm):
    """2x2 focal-plane panels for differential beam metrics (Pol-X minus Pol-Y)."""
    bd = results["beam_data"]

    # Wrap orientation difference to [-90, 90] (linear polarization periodicity)
    d_orient_raw = np.asarray(bd[4], dtype=float) - np.asarray(bd[5], dtype=float)
    d_orient     = (d_orient_raw + 90.0) % 180.0 - 90.0

    panels = [
        # (values, title, cbar_label, symmetric, cmap)
        (np.asarray(bd[6],  dtype=float),
         "dBeam centre [arcmin]", "arcmin", False, "viridis"),
        (np.asarray(bd[0],  dtype=float) - np.asarray(bd[2], dtype=float),
         "dFWHM major [arcmin]",  "arcmin", True,  "RdBu_r"),
        (np.asarray(bd[7],  dtype=float) - np.asarray(bd[8], dtype=float),
         "dEllipticity",          "",       True,  "RdBu_r"),
        (d_orient,
         "dOrientation [deg]",    "deg",    True,  "RdBu_r"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    fig.suptitle("Differential beam mismatch between polarization channels",
                 fontsize=_FS_TITLE + 1)

    for ax, (vals, title, unit, sym, cmap) in zip(axes.ravel(), panels):
        _focal_scatter(ax, x_mm, y_mm, vals, title=title,
                       cbar_label=unit if unit else title,
                       cmap=cmap, symmetric=sym)
        _draw_focal_boundary(ax, x_mm, y_mm)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 5: Representative Mueller beam maps (one figure per feed)
# IDL reference: per-feed 4-panel beam map figures
# ---------------------------------------------------------------------------

def _select_representative_feeds(results, x_mm, y_mm):
    """Select four representative feed indices and labels.

    Returns (indices, labels) with duplicates removed while preserving order.
    """
    mm = results["mueller_matrices"]
    psi_deg = compute_pol_rotation(results)

    r_sq = x_mm ** 2 + y_mm ** 2
    idx_centre = int(np.nanargmin(np.where(np.isfinite(r_sq), r_sq, np.inf)))
    idx_edge   = int(np.nanargmax(np.where(np.isfinite(r_sq), r_sq, -np.inf)))

    psi_abs = np.abs(psi_deg)
    idx_psi = int(np.nanargmax(np.where(np.isfinite(psi_abs), psi_abs, -np.inf)))

    QT = np.asarray(mm[_IDX["QT"]], dtype=float)
    UT = np.asarray(mm[_IDX["UT"]], dtype=float)
    tp = np.sqrt(QT ** 2 + UT ** 2)
    idx_tp = int(np.nanargmax(np.where(np.isfinite(tp), tp, -np.inf)))

    seen, labels = [], []
    for idx, label in [
        (idx_centre, "central feed"),
        (idx_edge,   "edge feed"),
        (idx_psi,    "max |pol rotation|"),
        (idx_tp,     "max |T→P leakage|"),
    ]:
        if idx not in seen:
            seen.append(idx)
            labels.append(label)
    return seen, labels


def fig_representative_beam_maps(results, x_mm, y_mm):
    """One 2x2-panel figure per representative feed.

    Shows TT, QT, UT, QU for each selected feed.
    Returns a list of Figure objects (one per feed).
    IDL reference: per-feed 4-panel beam map figures.
    """
    if "mueller_maps" not in results.files if hasattr(results, "files") \
            else "mueller_maps" not in results:
        print("  [make_diagnostic_plots] 'mueller_maps' not found in results; "
              "skipping Figure 5.")
        return []

    names_list   = _names(results)
    u_rad        = np.asarray(results["u"], dtype=float)
    v_rad        = np.asarray(results["v"], dtype=float)
    mueller_maps = np.asarray(results["mueller_maps"], dtype=float)

    u_arcmin = u_rad * RAD2ARCMIN
    v_arcmin = v_rad * RAD2ARCMIN
    extent   = [u_arcmin[0], u_arcmin[-1], v_arcmin[0], v_arcmin[-1]]

    feed_indices, feed_labels = _select_representative_feeds(results, x_mm, y_mm)
    figs = []

    for fidx, flabel in zip(feed_indices, feed_labels):
        name = names_list[fidx] if fidx < len(names_list) else f"feed_{fidx}"
        xf, yf = x_mm[fidx], y_mm[fidx]
        if np.isfinite(xf) and np.isfinite(yf):
            pos_str = f"x = {xf:.1f} mm,  y = {yf:.1f} mm"
        else:
            pos_str = "position unknown"

        panels = [
            # (element name, flat Mueller index, color style)
            ("TT", _IDX["TT"], "sequential"),
            ("QT", _IDX["QT"], "diverging"),
            ("UT", _IDX["UT"], "diverging"),
            ("QU", _IDX["QU"], "diverging"),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(10, 9))
        fig.suptitle(
            f"Mueller beam maps:  {name}  ({flabel})\n{pos_str}",
            fontsize=_FS_TITLE, y=1.01,
        )

        for ax, (elem, midx, style) in zip(axes.ravel(), panels):
            bmap = mueller_maps[:, :, midx, fidx]   # shape (nu, nv)
            peak = float(np.nanmax(np.abs(bmap)))
            if peak == 0 or not np.isfinite(peak):
                ax.set_title(f"{elem}  (no data)", fontsize=_FS_TITLE)
                continue

            norm_map = bmap / peak   # normalised to peak

            if style == "sequential":
                im = ax.imshow(
                    norm_map.T, origin="lower", extent=extent,
                    cmap="inferno", aspect="equal", vmin=0, vmax=1,
                )
                # Contours at 10 % and 50 % of peak
                ax.contour(
                    u_arcmin, v_arcmin, norm_map.T,
                    levels=[0.1, 0.5], colors="white",
                    linewidths=[0.6, 1.0], linestyles=["-", "--"],
                )
            else:
                im = ax.imshow(
                    norm_map.T, origin="lower", extent=extent,
                    cmap="RdBu_r", aspect="equal", vmin=-1, vmax=1,
                )

            cb = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
            cb.set_label("normalised response", fontsize=_FS_CBAR - 1)
            cb.ax.tick_params(labelsize=_FS_TICK - 1)
            ax.set_title(elem, fontsize=_FS_TITLE)
            ax.set_xlabel("u [arcmin]", fontsize=_FS_LABEL - 1)
            ax.set_ylabel("v [arcmin]", fontsize=_FS_LABEL - 1)
            ax.tick_params(labelsize=_FS_TICK - 1)

        fig.tight_layout()
        figs.append(fig)

    return figs


# ---------------------------------------------------------------------------
# Leakage helper  (matches IDL P_mag / P_angle from S1 = MM ## [1,0,0,0])
# ---------------------------------------------------------------------------

def compute_leakage(results):
    """Per-feed T→pol leakage fraction and orientation [deg].

    Matches the IDL quantities:
        S1 = MM ## [1, 0, 0, 0]   (apply T-only input)
        P_mag   = sqrt(QT² + UT²) / TT   (leakage fraction, dimensionless)
        P_angle = atan2(UT, QT) / 2       (leakage orientation [rad])

    Returns
    -------
    leakage_fraction     : ndarray, shape (n_feeds,)  — dimensionless
    leakage_orientation_deg : ndarray, shape (n_feeds,)  — degrees
    """
    mm = results["mueller_matrices"]
    TT = np.asarray(mm[_IDX["TT"]], dtype=float)
    QT = np.asarray(mm[_IDX["QT"]], dtype=float)
    UT = np.asarray(mm[_IDX["UT"]], dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(TT != 0, np.sqrt(QT**2 + UT**2) / TT, np.nan)
    orient_deg = np.degrees(np.arctan2(UT, QT)) / 2.0
    return frac, orient_deg


# ---------------------------------------------------------------------------
# IDL-style focal-plane plot (second version — closer to IDL visual style)
# ---------------------------------------------------------------------------

def plot_polarization_rotation_focal_plane(
    feed_x_mm,
    feed_y_mm,
    pol_rotation_deg,
    leakage_fraction=None,
    leakage_orientation_deg=None,
    aperture_radius_mm=None,
    aperture_x_radius_mm=None,
    aperture_y_radius_mm=None,
    output_path=None,
    ax=None,
    title="Focal-plane polarization rotation and leakage",
):
    """IDL-style focal-plane polarization-rotation diagnostic plot.

    Each feed is drawn as a circle at its physical focal-plane position.
    Circle color encodes the signed polarization-angle rotation (RdBu_r,
    positive = CW = red per IDL convention).  Circle size scales with the
    rotation magnitude.  Optional grey line segments show the T→pol leakage
    vector (length ∝ leakage %, orientation = leakage angle).

    IDL reference: the CW/CCW rotation + pseudo-vector page of array_map.ps.

    Parameters
    ----------
    feed_x_mm, feed_y_mm : array-like [mm]
        Focal-plane coordinates of each feed.
    pol_rotation_deg : array-like [deg]
        Signed induced polarization-angle rotation.  Positive = CW = red.
    leakage_fraction : array-like, optional
        Dimensionless T→pol leakage amplitude (e.g. 5e-4).
    leakage_orientation_deg : array-like, optional
        Orientation of the leakage vector [deg].
    aperture_radius_mm : float, optional
        Circular aperture radius.  Draws a boundary circle.
    aperture_x_radius_mm, aperture_y_radius_mm : float, optional
        Elliptical aperture semi-axes.
    output_path : str or Path, optional
        If given, saves the figure here (dpi=300, bbox_inches='tight').
    ax : matplotlib Axes, optional
        If given, draws into this axes rather than creating a new figure.
    title : str
        Figure title.

    Returns
    -------
    fig, ax
    """
    feed_x_mm         = np.asarray(feed_x_mm,         dtype=float)
    feed_y_mm         = np.asarray(feed_y_mm,         dtype=float)
    pol_rotation_deg  = np.asarray(pol_rotation_deg,  dtype=float)

    # ---- Create figure / axes ----
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7.2, 6.4), constrained_layout=True)
    else:
        fig = ax.figure

    # ---- Rotation colormap (RdBu_r: positive → red = CW, IDL convention) ----
    valid = np.isfinite(pol_rotation_deg) & np.isfinite(feed_x_mm) & np.isfinite(feed_y_mm)
    if valid.any():
        rot_lim = max(float(np.nanpercentile(np.abs(pol_rotation_deg[valid]), 98)), 1e-6)
    else:
        rot_lim = 1.0
    from matplotlib.colors import TwoSlopeNorm
    norm_rot = TwoSlopeNorm(vcenter=0.0, vmin=-rot_lim, vmax=rot_lim)
    cmap_rot = "RdBu_r"

    # ---- Marker size: area ∝ rotation² with a reference at 0.2 deg ----
    _ROT_REF_DEG  = 0.2
    _S_MIN        = 2.0
    _S_SCALE      = 50.0   # area per deg² above S_MIN (tune for visual clarity)
    rot_abs = np.abs(pol_rotation_deg)
    sizes = np.where(valid,
                     np.clip(_S_MIN + (rot_abs / _ROT_REF_DEG)**2 * _S_SCALE,
                             _S_MIN, 1400.0),
                     0.0)

    # ---- Draw circles (edge color only — fill left transparent so leakage
    #      sticks remain visible beneath the circles) ----
    if valid.any():
        from matplotlib.cm import ScalarMappable
        edge_colors = plt.get_cmap(cmap_rot)(norm_rot(pol_rotation_deg[valid]))
        ax.scatter(
            feed_x_mm[valid], feed_y_mm[valid],
            s=sizes[valid],
            facecolors="none",
            edgecolors=edge_colors,
            linewidths=1.2,
            zorder=4,
        )
        sm = ScalarMappable(norm=norm_rot, cmap=cmap_rot)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label("polarization rotation [deg]", fontsize=_FS_CBAR)
        cbar.ax.tick_params(labelsize=_FS_TICK)

    # ---- Leakage sticks ----
    _STICK_SCALE = 300.0   # mm per leakage-percent; tune to match IDL visual scale
    _REF_PERCENT = 0.05    # reference leakage in percent
    has_leakage = (leakage_fraction is not None) and (leakage_orientation_deg is not None)
    if has_leakage:
        lf  = np.asarray(leakage_fraction,       dtype=float)
        lo  = np.asarray(leakage_orientation_deg, dtype=float)
        lp  = 100.0 * lf                          # convert to percent
        theta = np.deg2rad(lo)
        half_len = 0.5 * lp * _STICK_SCALE
        dx = half_len * np.cos(theta)
        dy = half_len * np.sin(theta)
        vstick = valid & np.isfinite(lf) & np.isfinite(lo)
        for xi, yi, dxi, dyi in zip(
                feed_x_mm[vstick], feed_y_mm[vstick], dx[vstick], dy[vstick]):
            ax.plot([xi - dxi, xi + dxi], [yi - dyi, yi + dyi],
                    color="0.35", lw=0.9, alpha=0.8, zorder=3, solid_capstyle="round")

    # ---- Aperture boundary ----
    theta_circ = np.linspace(0, 2 * np.pi, 360)
    if aperture_radius_mm is not None:
        r = aperture_radius_mm
        ax.plot(r * np.cos(theta_circ), r * np.sin(theta_circ),
                color="navy", lw=1.2, ls=(0, (5, 3, 1, 3)), zorder=1)
    elif aperture_x_radius_mm is not None and aperture_y_radius_mm is not None:
        ax.plot(aperture_x_radius_mm * np.cos(theta_circ),
                aperture_y_radius_mm * np.sin(theta_circ),
                color="navy", lw=1.2, ls=(0, (5, 3, 1, 3)), zorder=1)
    else:
        # Fall back: circle at 5% beyond outermost feed
        r_max = float(np.nanmax(np.sqrt(feed_x_mm**2 + feed_y_mm**2)))
        if np.isfinite(r_max) and r_max > 0:
            r = r_max * 1.07
            ax.plot(r * np.cos(theta_circ), r * np.sin(theta_circ),
                    color="navy", lw=1.0, ls=":", zorder=1)

    # ---- Reference axes ----
    lim = float(np.nanmax(np.sqrt(feed_x_mm**2 + feed_y_mm**2))) * 1.15
    if not (np.isfinite(lim) and lim > 0):
        lim = 200.0
    ax.axhline(0, color="0.6", lw=1.0, ls=(0, (5, 3, 1, 3)), zorder=0)
    ax.axvline(0, color="0.6", lw=1.0, ls=(0, (5, 3, 1, 3)), zorder=0)

    # ---- Scale annotations (lower-left corner, matching IDL layout) ----
    x0 = -lim * 0.88
    y0 = -lim * 0.88
    y1 = y0 + lim * 0.07

    # Rotation scale circle at 0.2 deg (open circle, matching feed style)
    s_ref = _S_MIN + (_ROT_REF_DEG / _ROT_REF_DEG)**2 * _S_SCALE
    ax.scatter([x0], [y1], s=s_ref,
               facecolors="none", edgecolors="0.35", linewidths=1.2, zorder=5)
    ax.text(x0 + lim * 0.08, y1,
            f"{_ROT_REF_DEG:.1f} deg", va="center", fontsize=_FS_ANNOT,
            color="0.2")

    # Leakage scale stick
    if has_leakage:
        ref_half = 0.5 * _REF_PERCENT * _STICK_SCALE
        ax.plot([x0 - ref_half, x0 + ref_half], [y0, y0],
                color="0.25", lw=1.5, zorder=5)
        ax.text(x0 + ref_half + lim * 0.04, y0,
                f"{_REF_PERCENT:.2f} %", va="center", fontsize=_FS_ANNOT,
                color="0.2")

    # ---- CW / CCW labels (lower-right, matching IDL) ----
    ax.text(0.96, 0.10, "CW",  color="darkred", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=_FS_ANNOT + 1, fontweight="bold")
    ax.text(0.96, 0.05, "CCW", color="navy",    transform=ax.transAxes,
            ha="right", va="bottom", fontsize=_FS_ANNOT + 1, fontweight="bold")

    # ---- Axes formatting ----
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("[mm]", fontsize=_FS_LABEL)
    ax.set_ylabel("[mm]", fontsize=_FS_LABEL)
    ax.set_title(title, fontsize=_FS_TITLE)
    ax.tick_params(labelsize=_FS_TICK, direction="in",
                   top=True, right=True, which="both")
    ax.minorticks_on()

    # ---- Diagnostics printout ----
    valid_rot = pol_rotation_deg[valid]
    print(f"  [pol-rotation focal-plane plot]")
    print(f"    feeds plotted : {valid.sum()}")
    if valid_rot.size > 0:
        print(f"    rotation [deg]: min={valid_rot.min():+.3f}  "
              f"median={np.median(valid_rot):+.3f}  max={valid_rot.max():+.3f}  "
              f"RMS={np.sqrt(np.mean(valid_rot**2)):.3f}  "
              f"95th|ψ|={np.percentile(np.abs(valid_rot),95):.3f}")
    if has_leakage:
        lp_v = 100.0 * lf[vstick]
        if lp_v.size > 0:
            print(f"    leakage [%]  : min={lp_v.min():.4f}  "
                  f"median={np.median(lp_v):.4f}  max={lp_v.max():.4f}")

    # ---- Save ----
    if output_path is not None:
        suffix = Path(output_path).suffix.lower()
        dpi = 150 if suffix == ".png" else 300
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"    saved → {output_path}")

    return fig, ax


# ---------------------------------------------------------------------------
# Diagnostics printout and JSON save
# ---------------------------------------------------------------------------

def _run_diagnostics(results, x_mm, y_mm, diag_path=None):
    """Print summary statistics; optionally write them to a JSON file."""
    mm      = results["mueller_matrices"]
    psi_deg = compute_pol_rotation(results)
    valid   = psi_deg[np.isfinite(psi_deg)]

    QT = np.asarray(mm[_IDX["QT"]], dtype=float)
    UT = np.asarray(mm[_IDX["UT"]], dtype=float)
    tp = np.sqrt(QT ** 2 + UT ** 2)
    bd = results["beam_data"]

    diag = {
        "n_feeds": int(len(psi_deg)),
        "pol_rotation_deg": {
            "min":    float(np.nanmin(valid))                         if len(valid) else None,
            "median": float(np.nanmedian(valid))                      if len(valid) else None,
            "max":    float(np.nanmax(valid))                         if len(valid) else None,
            "rms":    float(np.sqrt(np.nanmean(valid ** 2)))          if len(valid) else None,
        },
        "max_abs_QT":               float(np.nanmax(np.abs(QT))),
        "max_abs_UT":               float(np.nanmax(np.abs(UT))),
        "max_TP_leakage":           float(np.nanmax(tp)),
        "max_diff_beam_centre_arcmin": float(np.nanmax(np.asarray(bd[6], dtype=float))),
    }

    print("=" * 58)
    print("  Key IDL reproduction figures — diagnostics")
    print("=" * 58)
    print(f"  Feeds plotted              : {diag['n_feeds']}")
    pr = diag["pol_rotation_deg"]
    print(f"  Pol-rotation [deg]         :")
    print(f"    min    = {pr['min']:+.4f}")
    print(f"    median = {pr['median']:+.4f}")
    print(f"    max    = {pr['max']:+.4f}")
    print(f"    RMS    = {pr['rms']:.4f}")
    print(f"  max |QT|                   = {diag['max_abs_QT']:.4e}")
    print(f"  max |UT|                   = {diag['max_abs_UT']:.4e}")
    print(f"  max sqrt(QT²+UT²)         = {diag['max_TP_leakage']:.4e}")
    print(f"  max dBeam centre [arcmin]  = {diag['max_diff_beam_centre_arcmin']:.3f}")
    print("=" * 58)

    if diag_path:
        Path(diag_path).write_text(json.dumps(diag, indent=2))
        print(f"  Diagnostics saved -> {diag_path}")

    return diag


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def plot_diagnostic_figures(
    results,
    output_pdf=None,
    output_dir=None,
    *,
    show=False,
    horn_layout_path=None,
):
    """Generate publication-quality reproductions of the key IDL diagnostic plots.

    Parameters
    ----------
    results : path, str, numpy NpzFile, or dict
        Array-map results from array_map.py (.npz or already-loaded object).
    output_pdf : str or Path, optional
        Save all figures into a single multi-page PDF.
    output_dir : str or Path, optional
        Save each figure as an individual PNG in this directory (created if needed).
    show : bool
        Call plt.show() after building all figures.
    horn_layout_path : str or Path, optional
        Override path to the horn-layout .dat file for physical mm coordinates.

    Figures produced
    ----------------
    1  Focal-plane polarization-angle rotation map   (IDL CW/CCW map)
    2  Polarization-angle rotation histogram         (IDL histogram panel 3)
    3  Integrated Mueller leakage maps QT/UT/QU/UQ   (IDL Mueller focal-plane maps)
    4  Differential beam mismatch map                (IDL beam mismatch diagnostics)
    5+ Representative Mueller beam maps per feed     (IDL per-feed beam maps)
    """
    results = _load_results(results)
    x_mm, y_mm, names = _focal_plane_mm(results, horn_layout_path)

    # Default destination: beam-mapping/outputs/
    if output_pdf is None and output_dir is None:
        output_pdf = str(_OUTPUTS_DIR / "diagnostic_figures.pdf")
        print(f"  No output specified; defaulting to: {output_pdf}")

    print(f"Building key IDL reproduction figures for {len(names)} feeds ...")

    psi_deg = compute_pol_rotation(results)
    lk_frac, lk_orient = compute_leakage(results)
    r_max = float(np.nanmax(np.sqrt(x_mm**2 + y_mm**2)))
    ap_r  = r_max if (np.isfinite(r_max) and r_max > 0) else None

    f1 = fig_pol_rotation_map(results, x_mm, y_mm)
    # Figure 1b: IDL-style version with circles + optional leakage sticks
    f1b, _ = plot_polarization_rotation_focal_plane(
        feed_x_mm=x_mm, feed_y_mm=y_mm,
        pol_rotation_deg=psi_deg,
        leakage_fraction=lk_frac,
        leakage_orientation_deg=lk_orient,
        aperture_radius_mm=ap_r,
        title="Focal-plane polarization rotation and leakage",
    )
    f2 = fig_pol_rotation_hist(results)
    f3 = fig_mueller_leakage_maps(results, x_mm, y_mm)
    f4 = fig_differential_beam_mismatch(results, x_mm, y_mm)
    f5_list = fig_representative_beam_maps(results, x_mm, y_mm)

    all_figs = [f1, f1b, f2, f3, f4] + f5_list

    # Choose diagnostics save path
    diag_path = None
    if output_pdf:
        diag_path = str(Path(output_pdf).with_suffix(".json"))
    elif output_dir:
        diag_path = str(Path(output_dir) / "diagnostics.json")
    _run_diagnostics(results, x_mm, y_mm, diag_path=diag_path)

    # ---- Save PDF ----
    if output_pdf:
        out = Path(output_pdf)
        out.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(out) as pdf:
            for fig in all_figs:
                pdf.savefig(fig, bbox_inches="tight")
        print(f"  Saved PDF -> {out}  ({len(all_figs)} pages)")

    # ---- Save individual PNGs ----
    if output_dir:
        odir = Path(output_dir)
        odir.mkdir(parents=True, exist_ok=True)
        stems = (
            ["fig1a_pol_rotation_map",
             "fig1b_pol_rotation_idl_style",
             "fig2_pol_rotation_hist",
             "fig3_mueller_leakage",
             "fig4_diff_beam_mismatch"]
            + [f"fig5_beam_maps_{i + 1}" for i in range(len(f5_list))]
        )
        for fig, stem in zip(all_figs, stems):
            p = odir / f"{stem}.png"
            fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved {len(all_figs)} PNGs -> {odir}/")

    if show:
        plt.show()

    for fig in all_figs:
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("data",
                   help="Path to array_data.npz produced by array_map.py")
    p.add_argument("--output-pdf", default=None, metavar="FILE",
                   help="Save all figures to a single PDF")
    p.add_argument("--output-dir", default=None, metavar="DIR",
                   help="Save individual PNGs to this directory")
    p.add_argument("--horn-layout", default=None, metavar="FILE",
                   help="Override horn-layout .dat file path")
    p.add_argument("--show", action="store_true",
                   help="Call plt.show() after building figures")
    args = p.parse_args()

    plot_diagnostic_figures(
        args.data,
        output_pdf=args.output_pdf,
        output_dir=args.output_dir,
        show=args.show,
        horn_layout_path=args.horn_layout,
    )


if __name__ == "__main__":
    main()
