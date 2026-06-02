"""
plot_array.py -- Visualise array-wide beam and Mueller matrix properties.

Takes the .npz file produced by array_map.py and generates a multi-page PDF:

  Page 1   Horn layout in focal plane (mm)
  Page 2   Mueller scalar values -- all 16 elements as coloured scatter plots
  Page 3   Averaged Mueller beam maps -- 4x4 grid
  Page 4   Performance metrics -- FWHM, ellipticity, gain per feed
  Page 5   Orientation and gain
  Page 6   Differential metrics (pol-X minus pol-Y)
  Page 7   Histograms -- 4x4 grid over all 16 Mueller elements

Usage
-----
python plot_array.py --data array_data.npz [options]

Optional
--------
--out FILE       Output PDF path (default: array_plots.pdf)
--horn_layout    Path to horn_layout .dat file
                 (default: data/horn_layout/horn_layout_331.dat)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import CenteredNorm, Normalize

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.read_horn_layout import load_layout
from core.config import HORN_LAYOUT_FILE

STOKES = ["T", "Q", "U", "V"]
MUELLER_LABELS = [f"{r}{c}" for r in STOKES for c in STOKES]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _scatter_metric(ax, phi_x, phi_y, values, label, cmap="viridis",
                    symmetric=False, unit="", s=80):
    valid = np.isfinite(values)
    if not valid.any():
        ax.set_title(label + " (no data)", fontsize=9)
        return
    if symmetric:
        vmax = max(np.nanmax(np.abs(values[valid])), 1e-12)
        norm = CenteredNorm(halfrange=vmax)
        cmap = "RdBu_r"
    else:
        vmin, vmax = np.nanmin(values[valid]), np.nanmax(values[valid])
        norm = Normalize(vmin=vmin, vmax=vmax)
    sc = ax.scatter(phi_x[valid], phi_y[valid], c=values[valid],
                    cmap=cmap, norm=norm, s=s, edgecolors="none")
    cb = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label(unit, fontsize=7)
    ax.set_xlabel("phi_x (deg)", fontsize=8)
    ax.set_ylabel("phi_y (deg)", fontsize=8)
    ax.set_aspect("equal")
    ax.set_title(label, fontsize=9)
    ax.tick_params(labelsize=7)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_horn_layout(pdf, horn_df):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(horn_df["x_mm"], horn_df["y_mm"], s=20, c="steelblue",
               edgecolors="none", alpha=0.8)
    for _, row in horn_df.iloc[::30].iterrows():
        ax.text(row["x_mm"], row["y_mm"], row["name"], fontsize=5,
                ha="center", va="bottom", color="dimgray")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Horn Layout -- Focal Plane (v2, 331 feeds)")
    ax.set_aspect("equal")
    ax.grid(True, lw=0.3, alpha=0.4)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_mueller_scalars(pdf, phi_x, phi_y, mueller_matrices):
    fig, axes = plt.subplots(4, 4, figsize=(16, 16), constrained_layout=True)
    fig.suptitle("Integrated Mueller Matrix Elements per Feed", fontsize=13)
    for idx in range(16):
        row_m, col_m = divmod(idx, 4)
        ax = axes[row_m, col_m]
        values = mueller_matrices[idx]
        sym = (row_m != col_m)
        _scatter_metric(ax, phi_x, phi_y, values,
                        label=MUELLER_LABELS[idx], symmetric=sym,
                        unit="(normalised)")
    pdf.savefig(fig)
    plt.close(fig)


def page_mueller_maps_overlay(pdf, phi_x, phi_y, mueller_maps, u, v):
    fig, axes = plt.subplots(4, 4, figsize=(16, 16), constrained_layout=True)
    fig.suptitle("Mueller Beam Maps -- Array Average (contours at 10%, 50%)", fontsize=12)
    u_arcmin = u * 60 * 180 / np.pi
    v_arcmin = v * 60 * 180 / np.pi

    for idx in range(16):
        row_m, col_m = divmod(idx, 4)
        ax = axes[row_m, col_m]
        mean_map = mueller_maps[:, :, idx, :].mean(axis=2)
        peak = np.abs(mean_map).max()
        if peak == 0:
            ax.set_visible(False)
            continue

        norm_map = mean_map / peak
        if row_m == col_m:
            im = ax.imshow(norm_map.T, origin="lower",
                           extent=[u_arcmin[0], u_arcmin[-1],
                                   v_arcmin[0], v_arcmin[-1]],
                           cmap="inferno", aspect="equal",
                           vmin=0, vmax=1)
            ax.contour(u_arcmin, v_arcmin, norm_map.T,
                       levels=[0.1, 0.5], colors="white", linewidths=0.6)
        else:
            im = ax.imshow(norm_map.T, origin="lower",
                           extent=[u_arcmin[0], u_arcmin[-1],
                                   v_arcmin[0], v_arcmin[-1]],
                           cmap="RdBu_r", aspect="equal",
                           vmin=-1, vmax=1)

        ax.set_title(MUELLER_LABELS[idx], fontsize=9)
        ax.set_xlabel("u (arcmin)", fontsize=7)
        ax.set_ylabel("v (arcmin)", fontsize=7)
        ax.tick_params(labelsize=6)
        plt.colorbar(im, ax=ax, shrink=0.8)

    pdf.savefig(fig)
    plt.close(fig)


def page_performance_metrics(pdf, phi_x, phi_y, beam_data):
    fig, axes = plt.subplots(3, 3, figsize=(15, 14), constrained_layout=True)
    fig.suptitle("Array Beam Performance Metrics", fontsize=13)
    metrics = [
        (beam_data[0],  "FWHM major -- Pol-X",  "arcmin", False),
        (beam_data[2],  "FWHM major -- Pol-Y",  "arcmin", False),
        (beam_data[12], "FWHM major -- TT",      "arcmin", False),
        (beam_data[1],  "FWHM minor -- Pol-X",  "arcmin", False),
        (beam_data[3],  "FWHM minor -- Pol-Y",  "arcmin", False),
        (beam_data[13], "FWHM minor -- TT",      "arcmin", False),
        (beam_data[7],  "Ellipticity -- Pol-X", "",       False),
        (beam_data[8],  "Ellipticity -- Pol-Y", "",       False),
        (beam_data[14], "Ellipticity -- TT",     "",       False),
    ]
    for ax, (vals, lbl, unit, sym) in zip(axes.ravel(), metrics):
        _scatter_metric(ax, phi_x, phi_y, vals, lbl, symmetric=sym, unit=unit)
    pdf.savefig(fig)
    plt.close(fig)


def page_orientation_gain(pdf, phi_x, phi_y, beam_data):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    fig.suptitle("Beam Orientation and Gain", fontsize=13)
    metrics = [
        (beam_data[4],  "Orientation -- Pol-X", "deg",   True),
        (beam_data[5],  "Orientation -- Pol-Y", "deg",   True),
        (beam_data[11], "Orientation -- TT",    "deg",   True),
        (beam_data[9],  "Gain proxy -- Pol-X",  "dBi",   False),
        (beam_data[10], "Gain proxy -- Pol-Y",  "dBi",   False),
        (beam_data[6],  "Diff. beam centre",    "arcmin",False),
    ]
    for ax, (vals, lbl, unit, sym) in zip(axes.ravel(), metrics):
        _scatter_metric(ax, phi_x, phi_y, vals, lbl, symmetric=sym, unit=unit)
    pdf.savefig(fig)
    plt.close(fig)


def page_differential(pdf, phi_x, phi_y, beam_data):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    fig.suptitle("Differential Beam Metrics  (Pol-X minus Pol-Y)", fontsize=13)
    metrics = [
        (beam_data[0] - beam_data[2], "dFWHM major",    "arcmin", True),
        (beam_data[1] - beam_data[3], "dFWHM minor",    "arcmin", True),
        (beam_data[7] - beam_data[8], "dEllipticity",   "",       True),
        (beam_data[4] - beam_data[5], "dOrientation",   "deg",    True),
    ]
    for ax, (vals, lbl, unit, sym) in zip(axes.ravel(), metrics):
        _scatter_metric(ax, phi_x, phi_y, vals, lbl, symmetric=sym, unit=unit)
    pdf.savefig(fig)
    plt.close(fig)


def page_histograms(pdf, mueller_matrices):
    fig, axes = plt.subplots(4, 4, figsize=(16, 14), constrained_layout=True)
    fig.suptitle("Distribution of Integrated Mueller Matrix Elements", fontsize=13)
    for idx in range(16):
        row_m, col_m = divmod(idx, 4)
        ax = axes[row_m, col_m]
        vals = mueller_matrices[idx]
        valid = vals[np.isfinite(vals)]
        if len(valid) == 0:
            continue
        val_range = valid.max() - valid.min()
        if val_range == 0 or not np.isfinite(val_range):
            ax.axvline(valid[0], color="steelblue", lw=2)
            ax.set_xlim(valid[0] - 1, valid[0] + 1)
        else:
            n_bins = min(30, max(5, len(np.unique(np.round(valid, 8)))))
            ax.hist(valid, bins=n_bins, color="steelblue", edgecolor="white", lw=0.4)
        ax.set_title(MUELLER_LABELS[idx], fontsize=9)
        ax.set_xlabel("Value", fontsize=7)
        ax.set_ylabel("Count", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.axvline(valid.mean(), color="crimson", lw=1.2,
                   label=f"mean={valid.mean():.3g}")
        ax.legend(fontsize=6)
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_plot_array(args):
    data = np.load(args.data, allow_pickle=True)
    mueller_maps = data["mueller_maps"]
    mueller_matrices = data["mueller_matrices"]
    beam_data = data["beam_data"]
    u = data["u"]
    v = data["v"]

    phi_x = mueller_matrices[16]
    phi_y = mueller_matrices[17]

    hl_path = Path(args.horn_layout) if args.horn_layout else HORN_LAYOUT_FILE
    horn_df = None
    try:
        horn_df = load_layout(hl_path)
    except Exception as e:
        print(f"Warning: could not load horn layout ({e}). Skipping focal-plane page.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing plots -> {out_path}")

    with PdfPages(out_path) as pdf:
        if horn_df is not None:
            page_horn_layout(pdf, horn_df)
        page_mueller_scalars(pdf, phi_x, phi_y, mueller_matrices)
        page_mueller_maps_overlay(pdf, phi_x, phi_y, mueller_maps, u, v)
        page_performance_metrics(pdf, phi_x, phi_y, beam_data)
        page_orientation_gain(pdf, phi_x, phi_y, beam_data)
        page_differential(pdf, phi_x, phi_y, beam_data)
        page_histograms(pdf, mueller_matrices)

    print("Done.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="array_plots.pdf")
    p.add_argument("--horn_layout", default=None)
    args = p.parse_args()
    run_plot_array(args)


if __name__ == "__main__":
    main()
