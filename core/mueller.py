"""
Mueller matrix construction from GRASP E-field maps.

Two .grd files are needed per feed:
    pol1.grd  — x-pol drive: field components (E_xx, E_yx)
    pol2.grd  — y-pol drive: field components (E_xy, E_yy)

Together they form the 2×2 Jones matrix at each pixel:
    J = [[E_xx, E_xy],
         [E_yx, E_yy]]

The 4×4 Mueller map is computed per pixel via:
    M = A  @  kron(J, conj(J))  @  inv(A)

where A is the Jones-to-Stokes transformation:
    A = 0.5 * [[1,  0,  0,  1],
                [1,  0,  0, -1],
                [0,  1,  1,  0],
                [0, -i,  i,  0]]

The result is real-valued (imaginary parts cancel) and is normalised
so that the integrated TT element equals 1.

Usage
-----
from core.mueller import build_mueller_maps, integrate_element

mueller = build_mueller_maps(field_pol1, field_pol2, u, v)
# mueller.shape == (nu, nv, 4, 4)

tt_integral = integrate_element(mueller, 0, 0, u, v)
"""

import numpy as np


# Jones → Stokes transformation matrix (4×4, complex)
_A = 0.5 * np.array(
    [
        [1,  0,  0,  1],
        [1,  0,  0, -1],
        [0,  1,  1,  0],
        [0, -1j, 1j, 0],
    ],
    dtype=complex,
)
_A_INV = np.linalg.inv(_A)


def _jones_to_mueller_pixel(J: np.ndarray) -> np.ndarray:
    """Convert a single 2×2 complex Jones matrix to a 4×4 real Mueller matrix."""
    K = np.kron(J, np.conj(J))  # 4×4 complex
    M = _A @ K @ _A_INV
    return M.real


def build_mueller_maps(
    field_pol1: np.ndarray,
    field_pol2: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """Build normalised 4×4 Mueller maps from two polarisation E-field arrays.

    Parameters
    ----------
    field_pol1 : complex ndarray, shape (nu, nv, 2)
        E-field for x-pol (co-pol) drive. [:,:,0]=E_xx, [:,:,1]=E_yx
    field_pol2 : complex ndarray, shape (nu, nv, 2)
        E-field for y-pol drive. [:,:,0]=E_xy, [:,:,1]=E_yy
    u, v : 1D float arrays
        Coordinate axes in radians (only used for normalisation pixel area).

    Returns
    -------
    mueller : float ndarray, shape (nu, nv, 4, 4)
        Normalised Mueller maps. mueller[i, j, row, col] is the (row,col)
        Mueller element at pixel (i, j).
    """
    nu, nv = field_pol1.shape[:2]

    E_xx = field_pol1[:, :, 0]
    E_yx = field_pol1[:, :, 1]
    E_xy = field_pol2[:, :, 0]
    E_yy = field_pol2[:, :, 1]

    # Vectorised Kronecker product approach
    # J = [[E_xx, E_xy], [E_yx, E_yy]] → flattened: (E_xx, E_xy, E_yx, E_yy)
    # kron(J, J*) expanded analytically for speed instead of per-pixel loop

    # Build J rows as (nu*nv, 2, 2) tensor then batch-compute
    J = np.stack(
        [
            np.stack([E_xx, E_xy], axis=-1),
            np.stack([E_yx, E_yy], axis=-1),
        ],
        axis=-2,
    )  # shape (nu, nv, 2, 2)

    J_flat = J.reshape(-1, 2, 2)          # (N, 2, 2)
    Jc_flat = np.conj(J_flat)

    # Batched Kronecker product: kron(J, J*) for each pixel.
    # Standard definition: K[2i+k, 2j+l] = J[i,j] * J*[k,l]
    # Build tensor T[n, i, k, j, l] = J[n,i,j] * J*[n,k,l], then reshape:
    #   (N, 2, 2, 2, 2) -> (N, 4, 4)  merging (i,k)->row and (j,l)->col (C order)
    # NOTE: the previous implementation used the outer-product layout
    #   K[i*2+j, k*2+l] = J[i,j]*J*[k,l], which is a permutation of the true
    #   Kronecker product and produced incorrect Mueller matrices.
    K_flat = (J_flat[:, :, None, :, None] * Jc_flat[:, None, :, None, :]).reshape(
        -1, 4, 4
    )  # (N, 4, 4), K_flat[n, 2i+k, 2j+l] = J[n,i,j]*J*[n,k,l]

    # Apply A @ K @ A_inv for all pixels
    M_flat = _A[None] @ K_flat @ _A_INV[None]  # (N, 4, 4) complex
    M_flat = M_flat.real

    mueller = M_flat.reshape(nu, nv, 4, 4)

    # Normalise: divide by integrated TT element
    tt_sum = integrate_element(mueller, 0, 0, u, v)
    if tt_sum != 0.0:
        mueller = mueller / tt_sum

    return mueller


def integrate_element(
    mueller: np.ndarray,
    row: int,
    col: int,
    u: np.ndarray,
    v: np.ndarray,
) -> float:
    """Integrate one Mueller matrix element over solid angle.

    Parameters
    ----------
    mueller : ndarray, shape (nu, nv, 4, 4)
    row, col : int  — which Mueller element (0-indexed)
    u, v : 1D coordinate arrays in radians

    Returns
    -------
    float : sum(element) * du * dv  (solid angle integral in rad²)
    """
    du = (u[-1] - u[0]) / (len(u) - 1)
    dv = (v[-1] - v[0]) / (len(v) - 1)
    return float(np.sum(mueller[:, :, row, col]) * du * dv)
