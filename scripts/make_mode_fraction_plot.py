"""
mode_fraction.py -- Plot aperture efficiency (mode fraction) sweep.

Expects a directory of single_beam.py output subdirectories, one per aperture
diameter, each containing a beam_params.txt.  The subdirectory name must include
the diameter in mm (e.g. "D_5.0", "D_7.5", etc.).

Usage
-----
python mode_fraction.py --data_dir DIR [options]

Required
--------
--data_dir DIR   Directory containing per-aperture single_beam output folders.

Optional
--------
--out FILE    Output PDF (default: mode_fraction.pdf)
--n_feeds N   Number of feeds in array (default: 331)
--f_number F  F/# of the optical system (default: 2.4)
--freq_ghz F  Observing frequency in GHz (default: 150)
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).parent.parent))


def _parse_beam_params(txt_path):
    """Extract pol-X beam metrics from beam_params.txt."""
    result = {}
    section = None
    with open(txt_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("=== "):
                section = line.strip("= ").strip()
            elif section and "Pol-X" in section:
                m = re.match(r"(\w+.*?)\s+([-\d.eEnanNiIf+]+)$", line)
                if m:
                    try:
                        result[m.group(1).strip()] = float(m.group(2))
                    except ValueError:
                        pass
    return result


def load_sweep_data(data_dir: Path):
    diameters, fwhm_maj, fwhm_min, ellip, gain = [], [], [], [], []
    for subdir in sorted(data_dir.iterdir()):
        params_file = subdir / "beam_params.txt"
        if not params_file.exists():
            continue
        m = re.search(r"(\d+\.?\d*)", subdir.name)
        if not m:
            continue
        D_mm = float(m.group(1))
        params = _parse_beam_params(params_file)
        if not params:
            continue
        diameters.append(D_mm)
        fwhm_maj.append(params.get("fwhm_major_arcmin", np.nan))
        fwhm_min.append(params.get("fwhm_minor_arcmin", np.nan))
        ellip.append(params.get("ellipticity", np.nan))
        gain.append(params.get("gain_dBi", np.nan))

    if not diameters:
        return None
    idx = np.argsort(diameters)
    return (np.array(diameters)[idx], np.array(fwhm_maj)[idx],
            np.array(fwhm_min)[idx], np.array(ellip)[idx], np.array(gain)[idx])


def airy_fwhm_arcmin(D_mm, freq_ghz=150.0):
    lam_mm = 300.0 / freq_ghz
    return 1.028 * lam_mm / D_mm * (180.0 / np.pi * 60.0)


def run_mode_fraction(args):
    data_dir = Path(args.data_dir)
    result = load_sweep_data(data_dir)
    if result is None:
        print(f"No valid beam_params.txt files found under {data_dir}")
        sys.exit(1)

    D, fwhm_a, fwhm_b, ellip, gain = result
    n = len(D)
    print(f"Loaded {n} aperture diameters: {D.min():.1f} - {D.max():.1f} mm")

    D_ref = np.linspace(D.min() * 0.8, D.max() * 1.2, 200)
    airy_ref = np.array([airy_fwhm_arcmin(d, args.freq_ghz) for d in D_ref])

    omega_beam = (np.pi / (4 * np.log(2))) * np.radians(fwhm_a / 60) * np.radians(fwhm_b / 60)
    omega_ref = (np.pi / (4 * np.log(2))) * np.radians(airy_fwhm_arcmin(D.max(), args.freq_ghz) / 60) ** 2
    mode_frac = omega_beam / omega_ref if omega_ref > 0 else np.ones(n)

    N = args.n_feeds
    noise_M  = 1.0 / (mode_frac * np.sqrt(N))
    noise_MN = 1.0 / np.sqrt(mode_frac * N)
    noise_1  = 1.0 / np.sqrt(N)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pdf:
        fig, axes = plt.subplots(2, 4, figsize=(16, 9), constrained_layout=True)
        fig.suptitle(f"Mode Fraction / Aperture Efficiency Sweep  "
                     f"(N={N} feeds, {args.freq_ghz} GHz)", fontsize=12)

        axes[0, 0].plot(D, mode_frac, "o-", ms=4)
        axes[0, 0].set(xlabel="Aperture D (mm)", ylabel="Mode fraction",
                       title="Mode Fraction vs. Diameter")
        axes[0, 0].grid(True, lw=0.3)

        axes[0, 1].plot(D, noise_1 * np.ones(n), "k--", lw=1, label="1/sqrt(N)")
        axes[0, 1].plot(D, noise_M, "b-o", ms=4, label="1/(M*sqrt(N))")
        axes[0, 1].plot(D, noise_MN, "r-^", ms=4, label="1/sqrt(M*N)")
        axes[0, 1].set(xlabel="Aperture D (mm)", ylabel="Noise scaling",
                       title="Noise Scaling vs. Diameter")
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, lw=0.3)

        axes[0, 2].plot(D, fwhm_a, "o-", ms=4, label="FWHM major")
        axes[0, 2].plot(D, fwhm_b, "s-", ms=4, label="FWHM minor")
        axes[0, 2].plot(D_ref, airy_ref, "k--", lw=1, label="Airy FWHM")
        axes[0, 2].set(xlabel="Aperture D (mm)", ylabel="FWHM (arcmin)",
                       title="FWHM vs. Diameter")
        axes[0, 2].legend(fontsize=8)
        axes[0, 2].grid(True, lw=0.3)

        axes[0, 3].plot(D, gain, "o-", ms=4)
        axes[0, 3].set(xlabel="Aperture D (mm)", ylabel="Gain proxy (dBi)",
                       title="Gain vs. Diameter")
        axes[0, 3].grid(True, lw=0.3)

        axes[1, 0].plot(D, ellip, "o-", ms=4)
        axes[1, 0].axhline(0, color="gray", lw=0.8, ls="--")
        axes[1, 0].set(xlabel="Aperture D (mm)", ylabel="Ellipticity",
                       title="Ellipticity vs. Diameter")
        axes[1, 0].grid(True, lw=0.3)

        axes[1, 1].plot(fwhm_a, mode_frac, "o-", ms=4)
        axes[1, 1].set(xlabel="FWHM major (arcmin)", ylabel="Mode fraction",
                       title="FWHM vs. Mode Fraction")
        axes[1, 1].grid(True, lw=0.3)

        axes[1, 2].plot(mode_frac, gain, "o-", ms=4)
        axes[1, 2].set(xlabel="Mode fraction", ylabel="Gain proxy (dBi)",
                       title="Mode Fraction vs. Gain")
        axes[1, 2].grid(True, lw=0.3)

        fov_mm = 300.0
        n_feeds_est = (fov_mm / D) ** 2 * (np.pi / 4)
        axes[1, 3].plot(D, n_feeds_est, "o-", ms=4)
        axes[1, 3].axhline(N, color="r", lw=1, ls="--", label=f"N={N}")
        axes[1, 3].set(xlabel="Aperture D (mm)", ylabel="Est. # feeds in FOV",
                       title="Array Size vs. Diameter")
        axes[1, 3].legend(fontsize=8)
        axes[1, 3].grid(True, lw=0.3)

        pdf.savefig(fig)
        plt.close(fig)

    print(f"Mode fraction plot -> {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out", default="mode_fraction.pdf")
    p.add_argument("--n_feeds", type=int, default=331)
    p.add_argument("--f_number", type=float, default=2.4)
    p.add_argument("--freq_ghz", type=float, default=150.0)
    args = p.parse_args()
    run_mode_fraction(args)


if __name__ == "__main__":
    main()
