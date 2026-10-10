#!/usr/bin/env python3
"""Single-file verification of every numerical statement in the manuscript.

    python3 verify_all.py            # full run (seconds to about a minute, machine dependent)
    python3 verify_all.py --quick    # skips the 2D lattice PDE runs (Section 6.4)

Each check prints PASS/FAIL with the computed value; the exit code is 1 if any check fails.
Requires: numpy, scipy.  No other files are needed.
Notes: expected values (656, 68, 0.2901, ...) belong to the parameters of Table 1 of the paper; if P is
changed they must be changed too.  Many checks compare two quantities computed from the same parameters
(e.g. formula vs direct count) and are therefore consistency checks.  The equality case lambda = R (blocks
v, w at k = 0, where g = h = R exactly) is decided with the tolerance TOL = 1e-9.
"""
import sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

QUICK = "--quick" in sys.argv
PHI = (1 + 5 ** 0.5) / 2
R, DELTA = PHI ** 2, PHI ** 3
TOL = 1e-9                                   # tolerance for "lambda <= R" at the equality case

# ------------------------------------------------------------------ model (example (4) of the paper)
P = dict(d=0.1, u_star=PHI, b=20 - 10 * PHI, g=PHI ** 2, h=PHI ** 2, a0=0.06, a1=0.08, c1=0.5, c2=0.3)
U2 = P["u_star"] - P["b"] / 3.0
BLOCK = {"u": P["b"], "v": P["g"], "w": P["h"]}


def F1(u, p=P): return -3.0 * (u - p["u_star"]) * (u - (p["u_star"] - p["b"] / 3.0))
def rhs_local(x, p=P):
    u, v, w = x
    return np.array([F1(u, p), -p["g"] * v + p["a0"] + p["a1"] * u, -p["h"] * w + p["c1"] * u + p["c2"] * v])
def equilibrium(p=P):
    u = p["u_star"]; v = (p["a0"] + p["a1"] * u) / p["g"]; w = (p["c1"] * u + p["c2"] * v) / p["h"]
    return np.array([u, v, w])
X0 = equilibrium()


def jac_k(k, p=P, e=1e-6):
    J = np.zeros((3, 3))
    for j in range(3):
        s = np.zeros(3); s[j] = e
        J[:, j] = (rhs_local(X0 + s, p) - rhs_local(X0 - s, p)) / (2 * e)
    return J - p["d"] * k ** 2 * np.eye(3)


def classify(lam):
    return "visible" if lam <= R * (1 + TOL) else ("intermediate" if lam <= DELTA * (1 + TOL) else "singular")


# ------------------------------------------------------------------ lattice utilities
def lam_lattice(N):
    m = np.fft.fftfreq(N) * N
    s = np.sin(np.pi * m / N) ** 2
    return 4 * N * N * (s[:, None] + s[None, :]), m[:, None] ** 2 + m[None, :] ** 2

def N2(t):
    if t < 0: return 0
    r = int(np.floor(np.sqrt(t))) + 1
    k = np.arange(-r, r + 1)
    return int(np.sum((k[:, None] ** 2 + k[None, :] ** 2) <= t + 1e-12))


# ------------------------------------------------------------------ bookkeeping
RESULTS = []
def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    sys.stdout.flush()


# ================================================================== 6.1 linear spectrum
print("== 6.1 Linear spectrum ==")
err = max(np.max(np.abs(np.sort(-np.linalg.eigvals(jac_k(k)).real) - np.sort([P["b"], P["g"], P["h"]]) - P["d"] * k ** 2))
          for k in np.linspace(0, 12, 49))
check("J(k): numeric eigenvalues = b_j + d k^2", err < 1e-8, f"max error {err:.2e}")
check("J(0) is lower triangular", np.max(np.abs(np.triu(jac_k(0.0), 1))) == 0.0)
c0 = [classify(x) for x in np.sort(-np.linalg.eigvals(jac_k(0.0)).real)]
c1 = [classify(x) for x in np.sort(-np.linalg.eigvals(jac_k(1.0)).real)]
check("classes at k=0: visible, visible, intermediate", c0 == ["visible", "visible", "intermediate"], str(c0))
check("classes at k=1: all intermediate", c1 == ["intermediate"] * 3, str(c1))
kU, kV = np.sqrt((DELTA - P["b"]) / P["d"]), np.sqrt((DELTA - P["g"]) / P["d"])
check("crossing of delta: k = 2.041 (u), 4.022 (v,w)", abs(kU - 2.0406) < 5e-4 and abs(kV - 4.0225) < 5e-4, f"{kU:.4f}, {kV:.4f}")
Jd = jac_k(0.0) + P["g"] * np.eye(3)
check("Jordan block: eigenvalue -phi^2 has geometric multiplicity 1",
      np.linalg.matrix_rank(Jd[1:, 1:], tol=1e-9) == 1)

# ================================================================== 6.2 theorems
print("\n== 6.2 Theorem 1 (spectral gap) ==")
rng = np.random.default_rng(1)
viol = mism = 0
for N in (2, 3, 4, 5, 8, 16, 33, 64):
    Lam, m2 = lam_lattice(N); nz = m2 > 0
    viol += np.any(16 * m2[nz] > Lam[nz] + 1e-9) + np.any(Lam[nz] > 4 * np.pi ** 2 * m2[nz] + 1e-9)
    for _ in range(200):
        b = rng.uniform(0.05, 5, 3); d = rng.uniform(1e-4, 1)
        mism += abs((b[:, None, None] + d * Lam[None]).min() - b.min()) > 1e-9
check("16|m|^2 <= Lambda_m <= 4 pi^2 |m|^2 for all tested N", viol == 0, f"violations {viol}")
check("smallest decay rate = min_j b_j (1600 random draws)", mism == 0, f"mismatches {mism}")

print("\n== 6.2 Theorem 2 (counting; N chosen from N_0) ==")
def N0_of(tmax, eps):
    return max(2 * np.sqrt(tmax) + 1, np.pi * (tmax + 1) / np.sqrt(3 * eps), np.pi ** 2 * np.sqrt(tmax / 6))
for d in (2e-4, 2e-3, 2e-2):
    for name in ("v", "u"):
        b = BLOCK[name]
        tR, tD = (R - b) / (4 * np.pi ** 2 * d), (DELTA - b) / (4 * np.pi ** 2 * d)
        eps = min([np.floor(t) + 1 - t for t in (tR, tD) if t >= 0] + [1.0])
        N = int(np.ceil(N0_of(max(1.0, tD), eps))) + 1
        Lam, _ = lam_lattice(N)
        r = b + d * Lam.ravel()
        V = int(np.sum(r <= R * (1 + TOL))); I = int(np.sum((r > R * (1 + TOL)) & (r <= DELTA)))
        predV, predI = N2(tR), N2(tD) - N2(tR)
        main = (DELTA - max(R, b)) / (4 * np.pi * d)
        bound = np.pi * (np.sqrt(2 * max(tD, 0)) + np.sqrt(2 * max(tR, 0)) + 1)
        check(f"d={d:g} block {name}: direct count = N2(t_delta)-N2(t_R) at N={N} (N0 formula)",
              (V, I) == (predV, predI) and abs(I - main) <= bound,
              f"direct (V,I)=({V},{I}), formula ({predV},{predI}), |I-main|={abs(I-main):.2f} <= {bound:.1f}")
Lam, _ = lam_lattice(512); r = R + 2e-4 * Lam.ravel()
I512 = int(np.sum((r > R * (1 + TOL)) & (r <= DELTA)))
check("d=2e-4, v-block: count at N=512 is still 656 (below N_0 = 1.4e3, Remark 2)", I512 == 656, f"I={I512}")
big = []
for Nb in (1024, 2048, 3072):
    Lam, _ = lam_lattice(Nb); r = R + 2e-4 * Lam.ravel()
    big.append(int(np.sum((r > R * (1 + TOL)) & (r <= DELTA))))
check("d=2e-4, v-block: counts 656, 640, 640 at N=1024, 2048, 3072; formula value 640", big == [656, 640, 640] and N2((DELTA - R) / (4 * np.pi ** 2 * 2e-4)) - N2(0.0) == 640, str(big))
tD6 = (DELTA - R) / (4 * np.pi ** 2 * 2e-3)
check("Section 6.6 example: t_delta = 20.49, eps = 0.507, N_0 = 54.7 (about 55)", abs(tD6 - 20.4926) < 1e-3 and abs(N0_of(tD6, np.floor(tD6) + 1 - tD6) - 54.73) < 0.02, f"N0 = {N0_of(tD6, np.floor(tD6) + 1 - tD6):.2f}")

print("\n== 6.2 Theorem 3 (data collapse) ==")
u_ = (np.arange(3000) + 0.5) / 3000
s_ = np.sin(np.pi * u_) ** 2
Tref = np.sort((s_[:, None] + s_[None, :]).ravel())
q = lambda p: float(np.quantile(Tref, p))
check("median of T = 1; T_0.1 = 0.2901; T_0.9 = 1.7099; ratio 5.894",
      abs(q(.5) - 1) < 1e-3 and abs(q(.1) - 0.2901) < 2e-3 and abs(q(.9) - 1.7099) < 2e-3 and abs(q(.9) / q(.1) - 5.894) < 0.02,
      f"{q(.1):.4f}, {q(.5):.4f}, {q(.9):.4f}, {q(.9)/q(.1):.3f}")
# exact limit G(theta) = P(T > theta): T = sin^2(pi U) + sin^2(pi V), a convolution of two arcsine laws
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
def cdf_T(th):
    if th <= 0: return 0.0
    if th >= 2: return 1.0
    f = lambda a: (2 / np.pi) * np.arcsin(np.sqrt(min(max(th - np.sin(a) ** 2, 0.0), 1.0)))
    pts = [b for b in ([np.arcsin(np.sqrt(th))] if th < 1 else [np.arcsin(np.sqrt(th - 1))] if th > 1 else []) if 0 < b < np.pi / 2]
    return (2 / np.pi) * quad(f, 0, np.pi / 2, points=pts or None, epsabs=1e-13, epsrel=1e-13, limit=400)[0]
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    th_grid = np.unique(np.concatenate([np.linspace(0, 2, 2001), 1 + np.geomspace(1e-9, 0.2, 200), 1 - np.geomspace(1e-9, 0.2, 200)]))
    G_spl = CubicSpline(th_grid, np.array([1 - cdf_T(t) for t in th_grid]))
G_ex = lambda t: np.where(np.abs(t - 1) < 1e-9, 0.5, G_spl(np.clip(t, 0, 2)))
check("exact limit: G(1) = 1/2 (median of T)", abs(1 - cdf_T(1.0) - 0.5) < 1e-10)
cnt_ok, sup_ok, sups = True, True, []
for N in (16, 32, 64, 128, 256, 512):
    Lam, _ = lam_lattice(N); T = np.sort(np.round(Lam.ravel() / (4 * N * N), 12)); n = T.size
    cnt_ok &= (int(np.sum(T == 1.0)) == 2 * (N - 1)) and (int(np.sum(T > 1.0)) == N * N // 2 - (N - 1))
    u_, first = np.unique(T, return_index=True)
    right = (n - np.searchsorted(T, u_, side="right")) / n          # G_N(t)  = #{T > t}/N^2
    left = (n - first) / n                                          # G_N(t-) = #{T >= t}/N^2
    dev = float(np.max(np.maximum(np.abs(right - G_ex(u_)), np.abs(left - G_ex(u_)))))
    sups.append(dev); sup_ok &= abs(dev - (N - 1) / N ** 2) < 2e-6
check("T = 1 occurs at exactly 2(N-1) modes and #{T>1} = N^2/2 - (N-1) (N = 16..512)", cnt_ok)
check("sup_theta |G_N - G| = (N-1)/N^2 for N = 16..512 (same for all blocks); 0.0586 ... 0.0019", sup_ok,
      ", ".join(f"{e:.4f}" for e in sups))
N = 256; Lam, _ = lam_lattice(N); T256 = np.round(Lam.ravel() / (4 * N * N), 12)
for name in ("v", "u"):
    c = DELTA - BLOCK[name]
    S_at = lambda x: float(np.mean(T256 > c / (4 * x)))
    check(f"block {name} (N=256): S_N(x) = 0 for x <= c/8; S_N(c/4) = 1/2 - (N-1)/N^2; x_50 = c/4 = {c/4:.4f}",
          S_at(c / 8) == 0.0 and abs(S_at(c / 4) - (0.5 - (N - 1) / N ** 2)) < 1e-12, f"S_N(c/4) = {S_at(c/4):.5f}")

# ================================================================== 6.3 nonlinear, uniform
print("\n== 6.3 Nonlinear return, spatially uniform ==")
J0 = jac_k(0.0)
dist = lambda x: np.linalg.norm(x - X0)
cases = {"v,w": X0 * [1, 1.5, 0.7], "u +40%": X0 * [1.4, 1, 1], "u -60%": X0 * [0.4, 1, 1], "all": X0 * [1.4, 1.5, 0.7]}
ratios = {}
for name, xi in cases.items():
    sol = solve_ivp(lambda t, x: rhs_local(x), [0, 9], xi, rtol=1e-13, atol=1e-16, dense_output=True)
    ratios[name] = [dist(sol.sol(t)) / np.linalg.norm(expm(J0 * t) @ (xi - X0)) for t in (2, 4, 6, 8)]
sol_vw = solve_ivp(lambda t, x: rhs_local(x), [0, 9], cases["v,w"], rtol=1e-13, atol=1e-16, dense_output=True)
check("v,w perturbation: distance at t=8 is 3e-11 (Table 4 note)", abs(dist(sol_vw.sol(8.0)) - 3e-11) < 5e-12, f"{dist(sol_vw.sol(8.0)):.2e}")
check("v,w perturbation: nonlinear = linear (ratio 1)", np.max(np.abs(np.array(ratios["v,w"]) - 1)) < 1e-3, str(np.round(ratios["v,w"], 5)))
for name in ("u +40%", "u -60%"):
    rr = ratios[name]
    check(f"{name}: ratio changes by < 1e-3 (relative) between t=6 and t=8", abs(rr[-1] - rr[-2]) / rr[-1] < 1e-3, str(np.round(rr, 5)))
rr = ratios["all"]
check("all fields: ratio increases monotonically for t = 2..8 (Jordan block, slow)", all(np.diff(rr) > 0) and abs(rr[-1] - rr[-2]) < abs(rr[1] - rr[0]), str(np.round(rr, 5)))

# long-time ratios in deviation coordinates (no round-off floor); values quoted in Section 6.3
bu = P["b"]
def rhs_dev(t, y):
    return [-3 * y[0] * (y[0] + bu / 3), -P["g"] * y[1] + P["a1"] * y[0], -P["h"] * y[2] + P["c1"] * y[0] + P["c2"] * y[1]]
long_r = {}
for name, fac in (("u +40%", [1.4, 1, 1]), ("u -60%", [0.4, 1, 1]), ("all", [1.4, 1.5, 0.7])):
    y0 = X0 * np.array(fac) - X0
    sol = solve_ivp(rhs_dev, [0, 40], y0, rtol=1e-13, atol=1e-90, dense_output=True, method="DOP853")
    long_r[name] = [np.linalg.norm(sol.sol(t)) / np.linalg.norm(expm(J0 * t) @ y0) for t in (8, 32, 40)]
check("t = 40 ratios: u+40% 0.7307, u-60% 2.903, all 0.813 (all: 0.732 -> 0.805 -> 0.813, still increasing)",
      abs(long_r["u +40%"][2] - 0.7307) < 2e-4 and abs(long_r["u -60%"][2] - 2.903) < 2e-3 and abs(long_r["all"][2] - 0.813) < 1e-3
      and abs(long_r["all"][1] - 0.805) < 1e-3 and long_r["all"][0] < long_r["all"][1] < long_r["all"][2],
      str({k: np.round(v, 4).tolist() for k, v in long_r.items()}))

# ================================================================== 6.4 2D lattice
if not QUICK:
    print("\n== 6.4 Nonlinear return on the 2D lattice (N=32, d=0.01) ==")
    N, kap = 32, 0.01; hg = 1.0 / N
    X, Y = np.meshgrid(np.arange(N) * hg, np.arange(N) * hg, indexing="ij")
    lap = lambda A: (np.roll(A, 1, 0) + np.roll(A, -1, 0) + np.roll(A, 1, 1) + np.roll(A, -1, 1) - 4 * A) / hg ** 2
    def rhs2d(t, y):
        u, v, w = y.reshape(3, N, N)
        return np.concatenate([(F1(u) + kap * lap(u)).ravel(),
                               (-P["g"] * v + P["a0"] + P["a1"] * u + kap * lap(v)).ravel(),
                               (-P["h"] * w + P["c1"] * u + P["c2"] * v + kap * lap(w)).ravel()])
    mode = np.cos(2 * np.pi * X)
    base = np.concatenate([np.full(N * N, c) for c in X0])
    lam_fd = 4 * N ** 2 * np.sin(np.pi / N) ** 2
    A = jac_k(0.0) - kap * lam_fd * np.eye(3)
    check("linear rates of mode (1,0): 3.0116 (v,w) and 4.2132 (u)",
          np.allclose(np.sort(-np.linalg.eigvals(A).real), [3.01155, 3.01155, 4.21318], atol=2e-5))
    ts = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    def run(amp):
        dev0 = np.concatenate([a * mode.ravel() * x for a, x in zip(amp, X0)])
        sol = solve_ivp(rhs2d, [0, 6], base + dev0, t_eval=ts, rtol=1e-12, atol=1e-15)
        a0 = np.array(amp) * X0
        rat = [hg * np.linalg.norm(sol.y[:, i] - base) / (np.linalg.norm(expm(A * t) @ a0) * np.linalg.norm(mode) * hg) for i, t in enumerate(ts)]
        Y3 = sol.y.reshape(3, N, N, -1)
        mean = np.array([[Y3[c, :, :, i].mean() - X0[c] for c in range(3)] for i in range(len(ts))])
        mrat = [np.linalg.norm(mean[i]) / np.linalg.norm(expm(J0 * (ts[i] - ts[1])) @ mean[1]) for i in (2, 3, 4)]
        return np.array(rat), np.array(mrat)
    r1, _ = run((1e-3, 1e-3, 1e-3))
    check("small perturbation: ratio = 1 (t=2..6)", np.max(np.abs(r1 - 1)) < 1e-3, str(np.round(r1, 4)))
    r2, _ = run((0.0, 0.0, 0.5))
    check("large w-only perturbation: ratio = 1", np.max(np.abs(r2 - 1)) < 1e-3, str(np.round(r2, 4)))
    r3, m3 = run((0.4, 0.5, -0.3))
    check("large u,v,w perturbation: ratio grows (slowest k=0 mode)", all(np.diff(r3) > 0) and r3[-1] > 3, str(np.round(r3, 3)))
    check("... and the spatial mean follows the uniform linear flow (ratio 1)", np.max(np.abs(m3 - 1)) < 1e-3, str(np.round(m3, 4)))

# ================================================================== 6.5 basin
print("\n== 6.5 Basin of attraction along u ==")
def diverges(fac):
    xi = X0.copy(); xi[0] = fac * X0[0]
    ev = lambda t, x: x[0] + 1e3
    ev.terminal = True
    sol = solve_ivp(lambda t, x: rhs_local(x), [0, 200], xi, rtol=1e-10, atol=1e-12, events=ev)
    return sol.t_events[0].size > 0
lo, hi = 0.0, 1.0
for _ in range(40):
    mid = 0.5 * (lo + hi)
    lo, hi = (mid, hi) if diverges(mid) else (lo, mid)
check("bisection boundary = u_2/u_* = 0.213107", abs(hi - U2 / P["u_star"]) < 1e-5, f"{hi:.6f} vs {U2/P['u_star']:.6f}")
check("no divergence for f = 10, 100, 1000", not any(diverges(f) for f in (10, 100, 1000)))

# ================================================================== 6.6 closure model
print("\n== 6.6 Closure model and stabilization of the mode count ==")
MU = 1.0
fcl = lambda t, y: MU * (1 + 1 / y - y)
okb = okm = True; rates = []
for rho0 in (0.05, 0.5, 1.0, 3.0, 20.0):
    sol = solve_ivp(fcl, [0, 15], [rho0], rtol=1e-12, atol=1e-14, dense_output=True)
    t = np.linspace(0, 15, 3001); rho = sol.sol(t)[0]
    okb &= bool(np.all(np.abs(rho - PHI) <= abs(rho0 - PHI) * np.exp(-MU * t) + 1e-12))
    okm &= bool(np.all(np.diff(rho) * np.sign(PHI - rho0) >= -1e-12))
    if rho0 <= 1.0:
        ts_ = np.linspace(4, 9, 200); rates.append(-np.polyfit(ts_, np.log(np.abs(sol.sol(ts_)[0] - PHI)), 1)[0])
check("Lemma 2: |rho - phi| <= |rho_0 - phi| e^{-mu tau}", okb)
check("Lemma 2: rho is monotone", okm)
check("linear rate = mu (3 - phi) = 1.3820", np.max(np.abs(np.array(rates) - MU * (3 - PHI))) < 2e-4, str(np.round(rates, 5)))
check("mu_1 = phi^2/(3-phi) = 1 + 2/sqrt(5); mu_2 = phi mu_1",
      abs(PHI ** 2 / (3 - PHI) - (1 + 2 / 5 ** 0.5)) < 1e-12 and abs(PHI * PHI ** 2 / (3 - PHI) - 3.065248) < 1e-6)
d, N = 2e-3, 256
Lam, _ = lam_lattice(N); rt = np.sort((R + d * Lam).ravel())
cnt = lambda dl: int(np.searchsorted(rt, dl, side="right") - np.searchsorted(rt, R * (1 + TOL), side="right"))
tD = (PHI * R - R) / (4 * np.pi ** 2 * d)
limit = N2(tD) - N2(0.0)
for rho0 in (1.05, 1.2, 3.0):
    sol = solve_ivp(fcl, [0, 30], [rho0], rtol=1e-12, atol=1e-14, dense_output=True)
    tt = np.linspace(0, 30, 6001)
    ser = np.array([cnt(R * sol.sol(t)[0]) for t in tt])
    last = tt[np.nonzero(np.diff(ser))[0].max() + 1]
    mono = np.all(np.diff(ser) >= 0) if rho0 < PHI else np.all(np.diff(ser) <= 0)
    check(f"Corollary 1: I(tau) constant for large tau, rho_0={rho0}", ser[-1] == limit == 68 and mono and last < 5,
          f"final {ser[-1]}, formula {limit}, last change at tau = {last:.2f}")

mu_, S_, G0_ = 0.7, 2.3, 9.0
solG = solve_ivp(lambda t, g: [mu_ * (S_ + S_ * S_ / g[0] - g[0])], [0, 10], [G0_], rtol=1e-12, atol=1e-14, dense_output=True)
tt = np.linspace(0.1, 9.9, 50); hh = 1e-6
Rp = (solG.sol(tt + hh)[0] - solG.sol(tt - hh)[0]) / (2 * hh * S_)
Rr = solG.sol(tt)[0] / S_
resid = np.max(np.abs(Rp - mu_ * (1 + 1 / Rr - Rr)))
check("realization G'=mu(S+S^2/G-G): R=G/S solves the closure ODE (residual of RK45 dense output)", resid < 1e-8, f"max |R' - f(R)| = {resid:.2e}")

# ================================================================== summary
n_ok = sum(ok for _, ok in RESULTS)
print(f"\n{n_ok}/{len(RESULTS)} checks passed" + ("  (quick mode: Section 6.4 skipped)" if QUICK else ""))
sys.exit(0 if n_ok == len(RESULTS) else 1)
