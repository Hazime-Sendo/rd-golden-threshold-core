"""Numerical checks of Theorems 1-3 (decay rate of block j and mode m: b_j + d * Lambda_m)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))   # reduced_rd.py, lattice.py
import numpy as np
from lattice import lam_lattice, N2

PHI = (1 + 5 ** 0.5) / 2
R, DELTA = PHI ** 2, PHI ** 3
B = {"u": 20 - 10 * PHI, "v": PHI ** 2, "w": PHI ** 2}
TOL = 1e-9
rng = np.random.default_rng(1)

# ---------------- Theorem 1 -------------------------------------------------------------
print("== Theorem 1 ==")
lo = hi = -np.inf
mism = trials = 0
for N in (2, 3, 4, 5, 8, 16, 33, 64):
    Lam, m2 = lam_lattice(N)
    nz = m2 > 0
    lo = max(lo, np.max(16 * m2[nz] - Lam[nz]))              # should be <= 0
    hi = max(hi, np.max(Lam[nz] - 4 * np.pi ** 2 * m2[nz]))  # should be <= 0
    for _ in range(200):
        b = rng.uniform(0.05, 5, 3)
        d = rng.uniform(1e-4, 1)
        rates = (b[:, None, None] + d * Lam[None]).min()
        mism += abs(rates - b.min()) > 1e-9
        trials += 1
print(f"max(16|m|^2 - Lambda) = {lo:.3g} (<=0), max(Lambda - 4pi^2|m|^2) = {hi:.3g} (<=0); "
      f"min-rate mismatches {mism}/{trials}")

# ---------------- Theorem 2 -------------------------------------------------------------
print("\n== Theorem 2: intermediate counts, N2 formula vs direct count ==")
print(" d        block   t_R        t_delta   pred I  pred V | direct I,V at N=64,128,256,512 | main term | Gauss bound")
for d in (2e-4, 2e-3, 2e-2):
    for name in ("v", "u"):
        b = B[name]
        tR, tD = (R - b) / (4 * np.pi ** 2 * d), (DELTA - b) / (4 * np.pi ** 2 * d)
        predI, predV = N2(tD) - N2(tR), N2(tR)
        row = []
        for N in (64, 128, 256, 512):
            Lam, _ = lam_lattice(N)
            r = b + d * Lam.ravel()
            V = int(np.sum(r <= R * (1 + TOL)))
            I = int(np.sum((r > R * (1 + TOL)) & (r <= DELTA)))
            row.append(f"{I},{V}")
        main = (DELTA - max(R, b)) / (4 * np.pi * d)
        bound = np.pi * (np.sqrt(2 * max(tD, 0)) + np.sqrt(2 * max(tR, 0)) + 1)
        print(f" {d:<7g}  {name}   {tR:9.3f}  {tD:9.3f}  {predI:5d}  {predV:4d} | {' | '.join(row)} | {main:8.2f} | {bound:6.1f}")

# ---------------- Theorem 3 -------------------------------------------------------------
print("\n== Theorem 3: singular fraction collapse in x = d N^2 ==")
def T_samples(n=4000):
    u = (np.arange(n) + 0.5) / n
    s = np.sin(np.pi * u) ** 2
    return (s[:, None] + s[None, :]).ravel()                # T = mu/4 = sin^2(pi U) + sin^2(pi V)
Tref = np.sort(T_samples())
q = lambda p: float(np.quantile(Tref, p))
print(f"quantiles of T: 10% {q(.1):.4f}, 50% {q(.5):.4f}, 90% {q(.9):.4f}; T90/T10 = {q(.9)/q(.1):.4f}")
for name in ("v", "u"):
    c = DELTA - B[name]
    print(f" block {name}: c = delta - b = {c:.6f}; x_50 = c/4 = {c/4:.6f}; x_10 = {c/(4*q(.9)):.6f}; x_90 = {c/(4*q(.1)):.6f}")
xs = np.geomspace(0.01, 5, 200)
print(" exact sup over all x is (N-1)/N^2 (attained at x = c/4); see verify_all.py")
for name in ("v", "u"):
    c = DELTA - B[name]
    Sinf = np.array([np.mean(Tref > c / (4 * x)) for x in xs])   # limit curve S(x)
    out = []
    for N in (16, 32, 64, 128, 256):
        Lam, _ = lam_lattice(N)
        L = np.sort(Lam.ravel()) / N**2
        S = np.array([1 - np.searchsorted(L, c / x, side="right") / L.size for x in xs])
        out.append(f"N={N}: {np.max(np.abs(S - Sinf)):.4f}")
    print(f" block {name}: max_x |S_N(x) - S_inf(x)| on 200 log-spaced x (a lower bound, not the sup)  " + "  ".join(out))
