"""
array_map.py -- Batch-process all feeds in a maps directory.

For each pol1/pol2 .grd pair found, computes Mueller maps, integrates the
4x4 Mueller matrix, and characterises both co-pol beams and the TT beam.
Results are saved to a compressed .npz file for use by plot_array.py.

Usage
-----
python array_map.py --maps_dir DIR [options]

Required
--------
--maps_dir DIR   Directory containing *_pol1.grd and *_pol2.grd files.

Optional
--------
--out FILE       Output .npz file path (default: array_data.npz)
--horn_layout    Path to horn_layout .dat file
                 (default: data/horn_layout/horn_layout_331.dat)

Output .npz keys
----------------
mueller_maps     float32, shape (nu, nv, 16, n_feeds)
mueller_matrices float64, shape (18, n_feeds)
                 Rows 0-15: integrated Mueller elements (TT,TQ,...,VV).
                 Rows 16-17: sky offsets (phi_x_deg, phi_y_deg).
beam_data        float64, shape (17, n_feeds)
                 [0-1]   FWHM major/minor pol-X (arcmin)
                 [2-3]   FWHM major/minor pol-Y (arcmin)
                 [4]     Orientation pol-X (deg)
                 [5]     Orientation pol-Y (deg)
                 [6]     Differential beam centre |dr| (arcmin)
                 [7-8]   Ellipticity pol-X, pol-Y
                 [9-10]  Gain pol-X, pol-Y (dBi proxy)
                 [11]    TT orientation (deg)
                 [12-13] TT FWHM major/minor (arcmin)
                 [14]    TT ellipticity
                 [15-16] Eccentricity pol-X, pol-Y
u, v             float64, coordinate axes in radians
names            str array, feed names extracted from filenames
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.read_grd import read_grd
from core.mueller import build_mueller_maps, integrate_element
from core.beam_utils import characterize_beam, power_beam
from core.read_horn_layout import load_layout
from core.config import HORN_LAYOUT_FILE


def _extract_feed_name(pol1_path: Path) -> str:
    """Extract feed name from filename like 'beam_map_331F1_pol1.grd' -> '331F1'."""
    stem = pol1_path.stem
    parts = stem.split("_")
    name_parts = [p for p in parts if p not in ("beam", "map", "pol1", "pol2")]
    return "_".join(name_parts) if name_parts else stem


def run_array_map(args):
    maps_dir = Path(args.maps_dir)
    pol1_files = sorted(maps_dir.glob("*_pol1.grd"))

    if not pol1_files:
        print(f"No *_pol1.grd files found in {maps_dir}")
        sys.exit(1)

    hl_path = Path(args.horn_layout) if args.horn_layout else HORN_LAYOUT_FILE
    try:
        horn_df = load_layout(hl_path)
        horn_lookup = {row["name"]: (row["phi_x_deg"], row["phi_y_deg"])
                       for _, row in horn_df.iterrows()}
    except Exception as e:
        print(f"Warning: could not load horn layout ({e}). Sky offsets will be 0.")
        horn_lookup = {}

    n_feeds = len(pol1_files)
    print(f"Found {n_feeds} feed pair(s) in {maps_dir}")

    mueller_maps_list = []
    mueller_matrices = np.zeros((18, n_feeds))
    beam_data = np.zeros((17, n_feeds))
    names = []
    u_last = v_last = None

    for feed_idx, pol1_path in enumerate(pol1_files):
        pol2_path = Path(str(pol1_path).replace("_pol1.grd", "_pol2.grd"))
        feed_name = _extract_feed_name(pol1_path)
        names.append(feed_name)

        print(f"  [{feed_idx+1}/{n_feeds}] {feed_name}", end="  ", flush=True)

        if not pol2_path.exists():
            print(f"WARNING: {pol2_path.name} not found -- skipping")
            mueller_maps_list.append(None)
            continue

        u, v, field_pol1 = read_grd(str(pol1_path))
        _, _, field_pol2 = read_grd(str(pol2_path))
        u_last, v_last = u, v

        mueller = build_mueller_maps(field_pol1, field_pol2, u, v)
        mueller_maps_list.append(mueller.reshape(*mueller.shape[:2], 16).astype(np.float32))

        for r in range(4):
            for c in range(4):
                mueller_matrices[r * 4 + c, feed_idx] = integrate_element(
                    mueller, r, c, u, v
                )

        phi_x, phi_y = horn_lookup.get(feed_name, (0.0, 0.0))
        mueller_matrices[16, feed_idx] = phi_x
        mueller_matrices[17, feed_idx] = phi_y

        pb1 = power_beam(field_pol1)
        pb2 = power_beam(field_pol2)
        tt_map = mueller[:, :, 0, 0]

        ch_x = characterize_beam(pb1, u, v)
        ch_y = characterize_beam(pb2, u, v)
        ch_tt = characterize_beam(tt_map, u, v)

        beam_data[0, feed_idx] = ch_x["fwhm_major_arcmin"]
        beam_data[1, feed_idx] = ch_x["fwhm_minor_arcmin"]
        beam_data[2, feed_idx] = ch_y["fwhm_major_arcmin"]
        beam_data[3, feed_idx] = ch_y["fwhm_minor_arcmin"]
        beam_data[4, feed_idx] = ch_x["orientation_deg"]
        beam_data[5, feed_idx] = ch_y["orientation_deg"]

        dx = (ch_x["center_u"] - ch_y["center_u"]) * 60 * 180 / np.pi
        dy = (ch_x["center_v"] - ch_y["center_v"]) * 60 * 180 / np.pi
        beam_data[6, feed_idx] = np.sqrt(dx**2 + dy**2)

        beam_data[7, feed_idx] = ch_x["ellipticity"]
        beam_data[8, feed_idx] = ch_y["ellipticity"]
        beam_data[9, feed_idx] = ch_x["gain_dBi"]
        beam_data[10, feed_idx] = ch_y["gain_dBi"]
        beam_data[11, feed_idx] = ch_tt["orientation_deg"]
        beam_data[12, feed_idx] = ch_tt["fwhm_major_arcmin"]
        beam_data[13, feed_idx] = ch_tt["fwhm_minor_arcmin"]
        beam_data[14, feed_idx] = ch_tt["ellipticity"]
        beam_data[15, feed_idx] = ch_x["eccentricity"]
        beam_data[16, feed_idx] = ch_y["eccentricity"]

        fwhm_mean = (ch_x["fwhm_major_arcmin"] + ch_x["fwhm_minor_arcmin"]) / 2
        print(f"FWHM(x)={fwhm_mean:.2f}' ell(x)={ch_x['ellipticity']:.4f}")

    if u_last is None:
        print("ERROR: no feeds were successfully processed.")
        sys.exit(1)

    nu, nv = u_last.shape[0], v_last.shape[0]
    mueller_maps_arr = np.zeros((nu, nv, 16, n_feeds), dtype=np.float32)
    for i, mm in enumerate(mueller_maps_list):
        if mm is not None:
            mueller_maps_arr[:, :, :, i] = mm

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        mueller_maps=mueller_maps_arr,
        mueller_matrices=mueller_matrices,
        beam_data=beam_data,
        u=u_last,
        v=v_last,
        names=np.array(names),
    )
    print(f"\nSaved array data -> {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--maps_dir", required=True)
    p.add_argument("--out", default="array_data.npz")
    p.add_argument("--horn_layout", default=None)
    args = p.parse_args()
    run_array_map(args)


if __name__ == "__main__":
    main()
