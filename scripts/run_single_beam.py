"""
single_beam.py -- Analyse a single GRASP feed beam map.

Usage
-----
python single_beam.py --pol1 PATH --pol2 PATH [options]

Required
--------
--pol1  Path to the x-polarisation .grd file
--pol2  Path to the y-polarisation .grd file

Optional
--------
--out DIR         Output directory (default: ./output_single/)
--realizations N  Number of CMB sky realisations to convolve (default: 0)
--bmodes          Include B-mode signal in CMB realisations
--r_tensor FLOAT  Tensor-to-scalar ratio for primordial B-modes (default: 0.1)
--mask            Mask beam beyond the first null before convolution
--cutoff_deg F    Mask radius in degrees (default: 3.0)

Outputs
-------
mueller_maps.pdf     4x4 grid of Mueller element maps (linear scale)
beam_cuts.pdf        E-plane and H-plane cuts for both polarisations
beam_params.txt      Beam characterisation metrics
mueller_matrix.npy   4x4 integrated Mueller matrix (unitless, normalised)
power_spectra.pdf    CMB power spectra (only if --realizations > 0)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import CenteredNorm

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.read_grd import read_grd
from core.mueller import build_mueller_maps, integrate_element
from core.beam_utils import characterize_beam, power_beam, mask_beam
from core.config import RAD2ARCMIN, RAD2DEG, DEG2RAD


STOKES = ["T", "Q", "U", "V"]
MUELLER_LABELS = [f"{r}{c}" for r in STOKES for c in STOKES]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _rad_to_arcmin_extent(u, v):
    return [u[0] * RAD2ARCMIN, u[-1] * RAD2ARCMIN,
            v[0] * RAD2ARCMIN, v[-1] * RAD2ARCMIN]


def plot_mueller_maps(mueller, u, v, out_pdf):
    """4x4 grid of Mueller element maps."""
    fig, axes = plt.subplots(4, 4, figsize=(14, 14), constrained_layout=True)
    extent = _rad_to_arcmin_extent(u, v)

    for idx in range(16):
        row_m, col_m = divmod(idx, 4)
        ax = axes[row_m, col_m]
        m = mueller[:, :, row_m, col_m]

        if row_m == col_m:
            im = ax.imshow(
                m.T, origin="lower", extent=extent,
                cmap="inferno", aspect="equal",
                vmin=0, vmax=m.max() if m.max() > 0 else 1,
            )
        else:
            vmax = max(np.abs(m).max(), 1e-12)
            im = ax.imshow(
                m.T, origin="lower", extent=extent,
                cmap="RdBu_r", aspect="equal",
                norm=CenteredNorm(halfrange=vmax),
            )

        ax.set_title(MUELLER_LABELS[idx], fontsize=9)
        ax.set_xlabel("u (arcmin)", fontsize=7)
        ax.set_ylabel("v (arcmin)", fontsize=7)
        ax.tick_params(labelsize=6)
        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

    fig.suptitle("Mueller Element Maps (normalised)", fontsize=12)
    out_pdf.savefig(fig)
    plt.close(fig)


def plot_beam_cuts(field_pol1, field_pol2, u, v, out_pdf):
    """E-plane and H-plane cuts for both polarisations."""
    pb1 = power_beam(field_pol1)
    pb2 = power_beam(field_pol2)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    u_arcmin = u * RAD2ARCMIN
    v_arcmin = v * RAD2ARCMIN

    for ax_row, (pb, label) in enumerate([(pb1, "Pol-X"), (pb2, "Pol-Y")]):
        peak_idx = np.unravel_index(pb.argmax(), pb.shape)
        peak = pb.max()

        e_cut = pb[:, peak_idx[1]] / peak
        h_cut = pb[peak_idx[0], :] / peak

        with np.errstate(divide="ignore"):
            e_db = 10 * np.log10(np.maximum(e_cut, 1e-10))
            h_db = 10 * np.log10(np.maximum(h_cut, 1e-10))

        ax_e = axes[ax_row, 0]
        ax_h = axes[ax_row, 1]

        ax_e.plot(u_arcmin, e_db)
        ax_e.axhline(-3, color="gray", lw=0.8, ls="--", label="-3 dB")
        ax_e.set_xlabel("u (arcmin)")
        ax_e.set_ylabel("Power (dB)")
        ax_e.set_title(f"{label} -- E-plane (u cut)")
        ax_e.set_ylim(-60, 2)
        ax_e.legend(fontsize=8)

        ax_h.plot(v_arcmin, h_db)
        ax_h.axhline(-3, color="gray", lw=0.8, ls="--", label="-3 dB")
        ax_h.set_xlabel("v (arcmin)")
        ax_h.set_ylabel("Power (dB)")
        ax_h.set_title(f"{label} -- H-plane (v cut)")
        ax_h.set_ylim(-60, 2)
        ax_h.legend(fontsize=8)

    fig.suptitle("Beam Cuts (normalised, dB scale)", fontsize=12)
    out_pdf.savefig(fig)
    plt.close(fig)


def plot_power_spectra(spectra_list, labels, out_pdf):
    """T, Q, U binned power spectra from multiple realisations."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    titles = ["TT", "QQ (EE proxy)", "UU (BB proxy)"]

    for ax, title, ci in zip(axes, titles, range(3)):
        for (ell, *ps), lbl in zip(spectra_list, labels):
            valid = np.isfinite(ell) & np.isfinite(ps[ci]) & (ps[ci] > 0)
            if valid.any():
                ax.loglog(ell[valid], ps[ci][valid], alpha=0.6, label=lbl)
        ax.set_xlabel("Multipole l")
        ax.set_ylabel("l(l+1)Cl / 2pi  [uK^2]")
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, which="both", lw=0.3, alpha=0.4)

    fig.suptitle("CMB Power Spectra from Mueller Convolution", fontsize=12)
    out_pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_single_beam(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading pol1: {args.pol1}")
    u, v, field_pol1 = read_grd(args.pol1)
    print(f"Reading pol2: {args.pol2}")
    _, _, field_pol2 = read_grd(args.pol2)

    print("Building Mueller maps...")
    mueller = build_mueller_maps(field_pol1, field_pol2, u, v)

    mm = np.array([
        [integrate_element(mueller, r, c, u, v) for c in range(4)]
        for r in range(4)
    ])
    np.save(out_dir / "mueller_matrix.npy", mm)
    print("Mueller matrix (integrated, normalised):")
    print(np.array2string(mm, precision=4, suppress_small=True))

    pb1 = power_beam(field_pol1)
    pb2 = power_beam(field_pol2)
    tt_map = mueller[:, :, 0, 0]

    ch_x = characterize_beam(pb1, u, v)
    ch_y = characterize_beam(pb2, u, v)
    ch_tt = characterize_beam(tt_map, u, v)

    params_path = out_dir / "beam_params.txt"
    with open(params_path, "w") as fh:
        for label, ch in [("Pol-X (co-pol)", ch_x), ("Pol-Y (co-pol)", ch_y),
                           ("TT Mueller", ch_tt)]:
            fh.write(f"=== {label} ===\n")
            for k, val in ch.items():
                fh.write(f"  {k:25s} {val:.6g}\n")
            fh.write("\n")
    print(f"Beam parameters -> {params_path}")

    mueller_pdf = out_dir / "mueller_maps.pdf"
    print(f"Plotting Mueller maps -> {mueller_pdf}")
    with PdfPages(mueller_pdf) as pdf:
        plot_mueller_maps(mueller, u, v, pdf)

    cuts_pdf = out_dir / "beam_cuts.pdf"
    print(f"Plotting beam cuts -> {cuts_pdf}")
    with PdfPages(cuts_pdf) as pdf:
        plot_beam_cuts(field_pol1, field_pol2, u, v, pdf)

    if args.realizations > 0:
        _run_cmb_convolution(args, mueller, u, v, out_dir)

    print("Done.")


def _run_cmb_convolution(args, mueller, u, v, out_dir):
    """Convolve Mueller beam with CMB realisations and plot power spectra."""
    from core.cmb_sim import cmb_realization, bin_spectra

    print(f"Running {args.realizations} CMB realisations...")
    nu, nv = mueller.shape[:2]
    pix_rad = (u[-1] - u[0]) / (nu - 1)

    mask = None
    if args.mask:
        cutoff_rad = args.cutoff_deg * DEG2RAD
        mask = mask_beam(u, v, mueller[:, :, 0, 0], cutoff_rad)

    all_spectra = []
    for i in range(args.realizations):
        print(f"  realisation {i+1}/{args.realizations}")
        try:
            _, _, sky = cmb_realization(
                nu, nv, pix_rad,
                include_bmodes=args.bmodes,
                r_tensor=args.r_tensor if args.bmodes else None,
            )
        except Exception as e:
            print(f"  WARNING: cmb_realization failed ({e}), skipping")
            continue

        T_sky, Q_sky, U_sky = sky[:, :, 0], sky[:, :, 1], sky[:, :, 2]

        TT = mueller[:, :, 0, 0].copy()
        QQ = mueller[:, :, 1, 1].copy()
        UU = mueller[:, :, 2, 2].copy()
        if mask is not None:
            TT[mask] = 0.0
            QQ[mask] = 0.0
            UU[mask] = 0.0

        # Pixel-area factor converts discrete sum to solid-angle integral
        def _convolve(beam_map, sky_map):
            return np.fft.ifft2(
                np.fft.fft2(beam_map) * np.fft.fft2(sky_map)
            ).real * pix_rad ** 2

        T_obs = _convolve(TT, T_sky)
        Q_obs = _convolve(QQ, Q_sky)
        U_obs = _convolve(UU, U_sky)

        # Flat-sky power spectrum (D_ell = l*(l+1)*C_l / 2pi)
        omega = pix_rad ** 2 * nu * nv
        ku = np.fft.fftfreq(nu, d=pix_rad)
        kv = np.fft.fftfreq(nv, d=pix_rad)
        KU, KV = np.meshgrid(ku, kv, indexing="ij")
        ell = 2 * np.pi * np.sqrt(KU ** 2 + KV ** 2)
        ell_flat = ell.ravel()

        def _ps_dl(m):
            fhat = np.fft.fft2(m)
            cl = (np.abs(fhat) ** 2 * omega / (nu * nv) ** 2).ravel()
            with np.errstate(divide="ignore", invalid="ignore"):
                dl = np.where(ell_flat > 1,
                              cl * ell_flat * (ell_flat + 1) / (2 * np.pi), 0.0)
            return dl

        n_bins, ell_r = 40, (70, 2500)
        e, tt = bin_spectra(ell_flat, _ps_dl(T_obs), n_bins, "log", ell_r)
        _, qq = bin_spectra(ell_flat, _ps_dl(Q_obs), n_bins, "log", ell_r)
        _, uu = bin_spectra(ell_flat, _ps_dl(U_obs), n_bins, "log", ell_r)
        all_spectra.append((e, tt, qq, uu))

    if not all_spectra:
        print("No valid realisations completed -- skipping power spectra plots.")
        return

    spec_pdf = out_dir / "power_spectra.pdf"
    print(f"Plotting power spectra -> {spec_pdf}")
    with PdfPages(spec_pdf) as pdf:
        plot_power_spectra(all_spectra,
                           [f"realiz. {i}" for i in range(len(all_spectra))],
                           pdf)

    spec_npz = out_dir / "spectra.npz"
    np.savez(spec_npz,
             ell=np.array([s[0] for s in all_spectra]),
             TT =np.array([s[1] for s in all_spectra]),
             QQ =np.array([s[2] for s in all_spectra]),
             UU =np.array([s[3] for s in all_spectra]))
    print(f"Spectra saved -> {spec_npz}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pol1", required=True)
    p.add_argument("--pol2", required=True)
    p.add_argument("--out", default="output_single")
    p.add_argument("--realizations", type=int, default=0)
    p.add_argument("--bmodes", action="store_true")
    p.add_argument("--r_tensor", type=float, default=0.1)
    p.add_argument("--mask", action="store_true")
    p.add_argument("--cutoff_deg", type=float, default=3.0)
    args = p.parse_args()
    run_single_beam(args)


if __name__ == "__main__":
    main()
