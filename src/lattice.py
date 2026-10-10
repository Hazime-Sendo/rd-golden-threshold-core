"""Lattice utilities: discrete Laplacian eigenvalues, Gauss-circle count N2."""
import numpy as np


def lam_lattice(N):
    """Eigenvalues Lambda_m of the (minus) 5-point Laplacian on the N x N torus of side 1,
    m = (m1, m2) with symmetric representatives; returns (Lambda, |m|^2) arrays of shape (N, N)."""
    m = np.fft.fftfreq(N) * N                    # integers; representatives -N/2 <= m < N/2 (equivalent to (-N/2, N/2] by symmetry)
    s = np.sin(np.pi * m / N) ** 2
    Lam = 4 * N * N * (s[:, None] + s[None, :])
    m2 = m[:, None] ** 2 + m[None, :] ** 2
    return Lam, m2


def N2(t):
    """N2(t) = #{m in Z^2 : |m|^2 <= t} (right-continuous)."""
    if t < 0:
        return 0
    r = int(np.floor(np.sqrt(t))) + 1
    k = np.arange(-r, r + 1)
    return int(np.sum((k[:, None] ** 2 + k[None, :] ** 2) <= t + 1e-12))
