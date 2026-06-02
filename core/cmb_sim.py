"""
CMB simulation utilities.

Functions
---------
read_models(models_dir)
    Load CAMB power spectra for all r values.

cmb_realization(nx, ny, pix_size_rad, include_bmodes=True, rng=None)
    Generate a single Gaussian-random CMB realisation (T, Q, U, E, B maps).

bin_spectra(ell, power, n_bins, spacing='log', ell_range=None)
    Bin a 1D power spectrum into n_bins logarithmic or linear bins.
"""

from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
from core.config import COSMO_MODELS_DIR, R_VALUES


# ---------------------------------------------------------------------------
# Power spectrum reading
# ---------------------------------------------------------------------------

def read_models(models_dir=None):
    """Read CAMB power spectra for all tensor-to-scalar ratio models.

    The files contain D_ell = ell*(ell+1)/(2*pi)*C_ell in units of μK².
    We return them as-is (D_ell) and handle the conversion in cmb_realization.

    Parameters
    ----------
    models_dir : Path or str, default = core.config.COSMO_MODELS_DIR

    Returns
    -------
    scalar : dict with keys 'ell', 'TT', 'EE', 'TE', 'BB'
        Scalar (adiabatic) power spectra from any r directory (same for all r).
    tensor : dict  r_value → {'ell', 'TT', 'EE', 'TE', 'BB'}
        Tensor spectra keyed by float r value.
    lensed : dict with keys 'ell', 'TT', 'EE', 'TE', 'BB'
        Lensed CMB power spectra (from any r directory; same for all r).
    """
    if models_dir is None:
        models_dir = COSMO_MODELS_DIR
    models_dir = Path(models_dir)

    def _load_cls(path, cols=(0, 1, 2, 3)):
        """Load columns from a CAMB-format .dat file, skipping header."""
        data = np.loadtxt(path, comments="#")
        ell = data[:, cols[0]].astype(float)
        return ell, data[:, cols[1]], data[:, cols[2]], data[:, cols[3]]

    # Use r_0.1 directory to load scalar and lensed (they are identical across r)
    ref_dir = models_dir / "r_0.1"

    ell_s, TT_s, EE_s, TE_s = _load_cls(ref_dir / "scalCls.dat")
    scalar = {"ell": ell_s, "TT": TT_s, "EE": EE_s, "TE": TE_s,
              "BB": np.zeros_like(TT_s)}

    ell_l, TT_l, EE_l, TE_l = _load_cls(ref_dir / "lensedCls.dat")
    # lensedCls columns: L TT EE BB TE
    ell_l2, TT_l2, EE_l2, BB_l2 = _load_lens(ref_dir / "lensedCls.dat")
    lensed = {"ell": ell_l2, "TT": TT_l2, "EE": EE_l2, "BB": BB_l2}

    tensor = {}
    for r in R_VALUES:
        r_dir = models_dir / f"r_{r}"
        ell_t, TT_t, EE_t, TE_t = _load_cls(r_dir / "tensCls.dat")
        # tensor lensedCls columns: L TT EE BB TE (need BB)
        _, _, _, BB_t = _load_lens(r_dir / "tensCls.dat")
        tensor[r] = {"ell": ell_t, "TT": TT_t, "EE": EE_t, "BB": BB_t}

    return scalar, tensor, lensed


def _load_lens(path):
    """Load ell, TT, EE, BB from a CAMB .dat file that has columns L TT EE BB TE."""
    data = np.loadtxt(path, comments="#")
    ell = data[:, 0]
    TT = data[:, 1]
    EE = data[:, 2]
    # Column 3 may be BB or TE depending on the file type
    col3 = data[:, 3]
    return ell, TT, EE, col3


# ---------------------------------------------------------------------------
# CMB realization
# ---------------------------------------------------------------------------

def cmb_realization(
    nx: int,
    ny: int,
    pix_size_rad: float,
    include_bmodes: bool = True,
    r_tensor: float = None,
    rng=None,
) -> tuple:
    """Generate a Gaussian-random CMB map realisation.

    Follows the algorithm in IDL cmb_realization.pro:
    - Creates 2D Fourier-space power from 1D CAMB spectra
    - Adds TE correlation via Cholesky-like decomposition
    - Converts E, B → Q, U
    - Returns real-space maps via inverse FFT

    Parameters
    ----------
    nx, ny : int — number of pixels (should be odd for symmetric grid)
    pix_size_rad : float — pixel size in radians
    include_bmodes : bool — include primordial/lensing B-modes
    r_tensor : float or None — if float, add tensor B-modes with this r value
    rng : numpy RandomState/Generator or None

    Returns
    -------
    x, y : 1D ndarrays of pixel coordinates in radians
    maps : ndarray, shape (nx, ny, 5)
        maps[:,:,0] = T
        maps[:,:,1] = Q
        maps[:,:,2] = U
        maps[:,:,3] = E
        maps[:,:,4] = B
    """
    if rng is None:
        rng = np.random.default_rng()

    # Map solid angle
    omega = pix_size_rad ** 2 * nx * ny  # total map area in rad²

    # Coordinate grids
    x = (np.arange(nx) - nx // 2) * pix_size_rad
    y = (np.arange(ny) - ny // 2) * pix_size_rad

    # Fourier-space wavenumber grids
    ku = np.fft.fftfreq(nx, d=pix_size_rad)  # cycles per radian
    kv = np.fft.fftfreq(ny, d=pix_size_rad)
    KU, KV = np.meshgrid(ku, kv, indexing="ij")

    # Multipole ell = 2π * |k|
    ELL = 2 * np.pi * np.sqrt(KU ** 2 + KV ** 2)
    PHI_K = np.arctan2(KV, KU)  # Fourier-plane angle for E→QU rotation

    # Load power spectra
    scalar, tensor, lensed = read_models()

    # Interpolate 1D C_ell onto 2D ell grid
    # Input D_ell → convert to C_ell = D_ell * 2π / (ell*(ell+1))
    def _interp_cl(spec_dict, key, ell_2d):
        ell1d = spec_dict["ell"]
        Dl = spec_dict[key]
        # avoid division by zero at ell=0,1
        with np.errstate(divide="ignore", invalid="ignore"):
            Cl = np.where(ell1d > 1, Dl * 2 * np.pi / (ell1d * (ell1d + 1)), 0.0)
        f = interp1d(ell1d, Cl, bounds_error=False, fill_value=0.0, kind="linear")
        return np.maximum(f(ell_2d.ravel()), 0.0).reshape(ell_2d.shape)

    T_ps = _interp_cl(scalar, "TT", ELL)
    E_ps = _interp_cl(scalar, "EE", ELL)
    TE_ps = _interp_cl(scalar, "TE", ELL)

    if include_bmodes:
        B_ps = _interp_cl(lensed, "BB", ELL)
        if r_tensor is not None and r_tensor in tensor:
            B_ps = B_ps + _interp_cl(tensor[r_tensor], "BB", ELL)
    else:
        B_ps = np.zeros_like(T_ps)

    # Normalization factor: variance per Fourier mode = C_ell * N^2 / omega
    norm = float(nx * ny) ** 2 / omega

    def _rand_field():
        return (rng.standard_normal((nx, ny)) + 1j * rng.standard_normal((nx, ny)))

    T_rand = _rand_field()
    E_rand = _rand_field()
    B_rand = _rand_field()

    # T field
    T_hat = np.sqrt(np.maximum(T_ps * norm, 0.0)) * T_rand

    # E with TE correlation: E = alpha * E_rand + beta * T_rand
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.where(T_ps > 0, TE_ps / np.sqrt(T_ps), 0.0)
        alpha_sq = np.maximum(E_ps - beta ** 2 * T_ps, 0.0)
    alpha = np.sqrt(alpha_sq * norm)
    beta_scaled = beta * np.sqrt(norm)
    E_hat = alpha * E_rand + beta_scaled * T_rand

    # B field
    B_hat = np.sqrt(np.maximum(B_ps * norm, 0.0)) * B_rand

    # Remove DC component
    T_hat[0, 0] = 0.0
    E_hat[0, 0] = 0.0
    B_hat[0, 0] = 0.0

    # E, B → Q, U  (spin-2 rotation)
    cos2phi = np.cos(2 * PHI_K)
    sin2phi = np.sin(2 * PHI_K)
    Q_hat = E_hat * cos2phi - B_hat * sin2phi
    U_hat = E_hat * sin2phi + B_hat * cos2phi

    # Inverse FFT to real space
    def _ifft_real(f_hat):
        return np.fft.ifft2(f_hat).real

    T_map = _ifft_real(T_hat)
    Q_map = _ifft_real(Q_hat)
    U_map = _ifft_real(U_hat)
    E_map = _ifft_real(E_hat)
    B_map = _ifft_real(B_hat)

    maps = np.stack([T_map, Q_map, U_map, E_map, B_map], axis=-1)

    return x, y, maps


# ---------------------------------------------------------------------------
# Power spectrum binning
# ---------------------------------------------------------------------------

def bin_spectra(
    ell: np.ndarray,
    power: np.ndarray,
    n_bins: int,
    spacing: str = "log",
    ell_range: tuple = None,
    min_pts: int = 3,
) -> tuple:
    """Bin a 1D power spectrum.

    Parameters
    ----------
    ell : 1D float array — multipole values
    power : 1D float array — power spectrum values
    n_bins : int — number of output bins
    spacing : 'log' or 'linear'
    ell_range : (ell_min, ell_max) or None (uses data range)
    min_pts : int — minimum points required to report a bin (else NaN)

    Returns
    -------
    ell_b : 1D array of bin-centre ell values
    pow_b : 1D array of averaged power per bin (NaN for empty bins)
    """
    if ell_range is None:
        ell_min, ell_max = ell.min(), ell.max()
    else:
        ell_min, ell_max = ell_range

    if spacing == "log":
        edges = np.exp(
            np.linspace(np.log(ell_min), np.log(ell_max), n_bins + 1)
        )
    else:
        edges = np.linspace(ell_min, ell_max, n_bins + 1)

    ell_b = np.full(n_bins, np.nan)
    pow_b = np.full(n_bins, np.nan)

    for i in range(n_bins):
        mask = (ell >= edges[i]) & (ell < edges[i + 1])
        if mask.sum() >= min_pts:
            ell_b[i] = ell[mask].mean()
            pow_b[i] = power[mask].mean()

    return ell_b, pow_b
