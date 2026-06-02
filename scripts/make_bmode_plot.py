"""
bmode_plot.py -- Compare B-mode contamination from beam systematics vs CMB signals.

Loads pre-computed power spectra and Mueller matrices from single_beam.py output
directories for two feeds, and plots them against CMB theoretical models.

The systematic error estimate follows the IDL formulation:
    B_sys(l) = E_ps * sin^2(dpsi) + 0.5 * (M_QT^2 + M_UT^2) * T_ps
where:
    dpsi = polarisation rotation angle from M_UQ element (Mueller[2,1])
    M_QT = Mueller[1,0]  (Q <- T leakage)
    M_UT = Mueller[2,0]  (U <- T leakage)

Usage
-----
python bmode_plot.py --feed1 DIR --feed2 DIR [options]

Required
--------
--feed1 DIR   Output dir from single_beam.py for first feed (centre)
--feed2 DIR   Output dir from single_beam.py for second feed (edge)

Optional
--------
--out FILE    Output PDF (default: bmode_plot.pdf)
--r_tensor F  Tensor-to-scalar ratio to highlight (default: 0.1)
--ell_max N   Max multipole for log plots (default: 2500)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.cmb_sim import read_models, bin_spectra


def systematic_bmode(mueller_matrix, scalar, lensed, ell_arr):
    """Estimate systematic B-mode D_ell from integrated Mueller matrix."""
    M_QT = mueller_matrix[1, 0]
    M_UT = mueller_matrix[2, 0]
    M_UQ = mueller_matrix[2, 1]
    delta_psi = 0.5 * np.arctan2(M_UQ, mueller_matrix[1, 1])

    def _dl(spec_dict, key):
        f = interp1d(spec_dict["ell"], spec_dict[key],
                     bounds_error=False, fill_value=0.0)
        return np.maximum(f(ell_arr), 0.0)

    T_dl = _dl(scalar, "TT")
    E_dl = _dl(scalar, "EE")

    return E_dl * np.sin(delta_psi) ** 2 + 0.5 * (M_QT ** 2 + M_UT ** 2) * T_dl


def run_bmode_plot(args):
    feed1_dir = Path(args.feed1)
    feed2_dir = Path(args.feed2)

    def _load_feed(d):
        mm_path = d / "mueller_matrix.npy"
        spec_path = d / "spectra.npz"
        mm = np.load(mm_path) if mm_path.exists() else np.eye(4)
        if spec_path.exists():
            raw = np.load(spec_path)
            # Each row is one realisation; zip into list of (ell, TT, QQ, UU)
            specs = list(zip(raw["ell"], raw["TT"], raw["QQ"], raw["UU"]))
        else:
            specs = None
        return mm, specs

    mm1, specs1 = _load_feed(feed1_dir)
    mm2, specs2 = _load_feed(feed2_dir)

    scalar, tensor, lensed = read_models()
    r_val = args.r_tensor

    ell_log = np.logspace(np.log10(70), np.log10(args.ell_max), 200)
    ell_lin = np.linspace(2, 400, 200)

    B_sys1_log = systematic_bmode(mm1, scalar, lensed, ell_log)
    B_sys2_log = systematic_bmode(mm2, scalar, lensed, ell_log)
    B_sys1_lin = systematic_bmode(mm1, scalar, lensed, ell_lin)
    B_sys2_lin = systematic_bmode(mm2, scalar, lensed, ell_lin)

    def _lensed_BB(ell):
        f = interp1d(lensed["ell"], lensed["BB"],
                     bounds_error=False, fill_value=0.0)
        return np.maximum(f(ell), 1e-10)

    def _tensor_BB(r, ell):
        if r not in tensor:
            return np.zeros_like(ell)
        f = interp1d(tensor[r]["ell"], tensor[r]["BB"],
                     bounds_error=False, fill_value=0.0)
        return np.maximum(f(ell), 1e-10)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pdf:

        # Page 1: log-log systematic vs CMB models
        fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
        fig.suptitle("B-mode Contamination: Systematic Error Estimate", fontsize=12)

        for ax, B_sys, mm, lbl in [
            (axes[0], B_sys1_log, mm1, f"Feed 1  ({feed1_dir.name})"),
            (axes[1], B_sys2_log, mm2, f"Feed 2  ({feed2_dir.name})"),
        ]:
            ax.loglog(ell_log, _lensed_BB(ell_log), "k-", lw=1.5,
                      label="Lensing BB")
            for r, ls in zip([0.001, 0.01, 0.1], [":", "--", "-."]):
                ax.loglog(ell_log, _tensor_BB(r, ell_log),
                          color="darkorange", ls=ls, lw=1.2,
                          label=f"Tensor r={r}")
            ax.loglog(ell_log, np.maximum(B_sys, 1e-10), "r-", lw=2.0,
                      label="B_sys estimate")
            ax.set_xlim(70, args.ell_max)
            ax.set_ylim(1e-8, 1e4)
            ax.set_xlabel("Multipole l", fontsize=10)
            ax.set_ylabel("l(l+1)Cl / 2pi  [uK^2]", fontsize=10)
            ax.set_title(lbl, fontsize=10)
            ax.legend(fontsize=7, ncol=2)
            ax.grid(True, which="both", lw=0.3, alpha=0.4)
            txt = (f"M_QT={mm[1,0]:.2e}  M_UT={mm[2,0]:.2e}\n"
                   f"M_UQ={mm[2,1]:.2e}")
            ax.text(0.02, 0.02, txt, transform=ax.transAxes,
                    fontsize=7, va="bottom", family="monospace",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: linear scale, low-l
        fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
        fig.suptitle("B-mode Contamination: Low-l (Linear Scale)", fontsize=12)

        for ax, B_sys, lbl in [
            (axes[0], B_sys1_lin, f"Feed 1  ({feed1_dir.name})"),
            (axes[1], B_sys2_lin, f"Feed 2  ({feed2_dir.name})"),
        ]:
            ax.plot(ell_lin, _lensed_BB(ell_lin), "k-", lw=1.5,
                    label="Lensing BB")
            for r, ls in zip([0.001, 0.01, 0.1], [":", "--", "-."]):
                ax.plot(ell_lin, _tensor_BB(r, ell_lin),
                        color="darkorange", ls=ls, lw=1.2,
                        label=f"Tensor r={r}")
            ax.plot(ell_lin, np.maximum(B_sys, 0), "r-", lw=2.0,
                    label="B_sys estimate")
            ax.set_xlim(0, 400)
            ax.set_xlabel("Multipole l", fontsize=10)
            ax.set_ylabel("l(l+1)Cl / 2pi  [uK^2]", fontsize=10)
            ax.set_title(lbl, fontsize=10)
            ax.legend(fontsize=7)
            ax.grid(True, lw=0.3, alpha=0.4)

        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: convolved spectra (if available)
        if specs1 is not None or specs2 is not None:
            fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
            fig.suptitle("B-mode Contamination: Convolved Spectra", fontsize=12)

            for ax, specs, lbl in [
                (axes[0], specs1, f"Feed 1  ({feed1_dir.name})"),
                (axes[1], specs2, f"Feed 2  ({feed2_dir.name})"),
            ]:
                if specs is None:
                    ax.text(0.5, 0.5, "No spectra available",
                            ha="center", transform=ax.transAxes)
                    continue
                ax.loglog(ell_log, _lensed_BB(ell_log), "k-", lw=1.5,
                          label="Lensing BB")
                ell_arr = None
                uu_stack = []
                for entry in specs:
                    ell_e, tt_e, qq_e, uu_e = entry
                    valid = np.isfinite(uu_e) & np.isfinite(ell_e) & (uu_e > 0)
                    if valid.any():
                        ell_arr = ell_e[valid]
                        uu_stack.append(uu_e[valid])
                if uu_stack and ell_arr is not None:
                    uu_mean = np.nanmean(np.array(uu_stack), axis=0)
                    ax.loglog(ell_arr, np.maximum(uu_mean, 1e-10),
                              "b-", lw=1.5, label="Convolved UU")
                ax.set_xlim(70, args.ell_max)
                ax.set_xlabel("Multipole l", fontsize=10)
                ax.set_ylabel("l(l+1)Cl / 2pi  [uK^2]", fontsize=10)
                ax.set_title(lbl, fontsize=10)
                ax.legend(fontsize=8)
                ax.grid(True, which="both", lw=0.3, alpha=0.4)

            pdf.savefig(fig)
            plt.close(fig)

    print(f"B-mode plot -> {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feed1", required=True)
    p.add_argument("--feed2", required=True)
    p.add_argument("--out", default="bmode_plot.pdf")
    p.add_argument("--r_tensor", type=float, default=0.1)
    p.add_argument("--ell_max", type=int, default=2500)
    args = p.parse_args()
    run_bmode_plot(args)


if __name__ == "__main__":
    main()
