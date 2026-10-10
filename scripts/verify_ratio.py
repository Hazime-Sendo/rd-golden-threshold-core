"""Closure model  rho' = mu (1 + 1/rho - rho): convergence to phi, local rate, and the
stabilisation of the intermediate-mode count I_b(tau) (Section 4-5 of the manuscript)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))   # reduced_rd.py, lattice.py
import numpy as np
from scipy.integrate import solve_ivp
from lattice import lam_lattice, N2

PHI = (1 + 5 ** 0.5) / 2
MU = 1.0
f = lambda t, y: MU * (1 + 1 / y - y)

print("== Lemma 2: |rho - phi| <= |rho0 - phi| exp(-mu tau) ==")
for rho0 in (0.05, 0.5, 1.0, 3.0, 20.0):
    sol = solve_ivp(f, [0, 15], [rho0], rtol=1e-12, atol=1e-14, dense_output=True)
    t = np.linspace(0, 15, 3001)
    u = np.abs(sol.sol(t)[0] - PHI)
    bound = abs(rho0 - PHI) * np.exp(-MU * t)
    viol = np.max(u - bound)
    ts = np.linspace(4, 9, 200)
    slope = -np.polyfit(ts, np.log(np.abs(sol.sol(ts)[0] - PHI)), 1)[0]
    mono = np.all(np.diff(sol.sol(t)[0]) * np.sign(PHI - rho0) >= -1e-12)
    resid = np.max(np.abs(f(0, sol.sol(t)) - np.gradient(sol.sol(t)[0], t)))
    print(f" rho0={rho0:5.2f}: max(|u|-bound) = {viol:.2e} (<=0), monotone: {mono}, "
          f"late rate {slope:.6f} vs mu(3-phi) = {MU*(3-PHI):.6f}")

print("\n== Corollary: I_b(tau) = #{modes with R < lambda <= delta(tau)} ==")
R = PHI ** 2
d, N = 2e-3, 256
Lam, _ = lam_lattice(N)
rates = np.sort((R + d * Lam).ravel())           # block with b = R
count = lambda delta: int(np.searchsorted(rates, delta, side="right") - np.searchsorted(rates, R * (1 + 1e-9), side="right"))
tD = (PHI * R - R) / (4 * np.pi ** 2 * d)
print(f" d={d}, N={N}: limit formula N2(t_delta) - N2(t_R) = {N2(tD) - N2(0)}, t_delta(inf) = {tD:.3f}")
for rho0 in (1.05, 1.2, 3.0):
    sol = solve_ivp(f, [0, 30], [rho0], rtol=1e-12, atol=1e-14, dense_output=True)
    taus = np.array([0, 0.5, 1, 2, 3, 5, 8, 12, 20, 30])
    Ib = [count(R * sol.sol(t)[0]) for t in taus]
    tt = np.linspace(0, 30, 6001)
    series = np.array([count(R * sol.sol(t)[0]) for t in tt])
    last_change = tt[np.max(np.nonzero(np.diff(series))[0]) + 1] if np.any(np.diff(series)) else 0.0
    print(f" rho0={rho0}: I_b at tau={list(taus)} -> {Ib}; last change at tau = {last_change:.2f}; final {series[-1]}")

print("\n== Realization G' = mu (S + S^2/G - G), R = G/S ==")
mu_, S_, G0_ = 0.7, 2.3, 9.0
solG = solve_ivp(lambda t, g: [mu_ * (S_ + S_ * S_ / g[0] - g[0])], [0, 10], [G0_], rtol=1e-12, atol=1e-14, dense_output=True)
tt = np.linspace(0.1, 9.9, 50); hh = 1e-6
Rp = (solG.sol(tt + hh)[0] - solG.sol(tt - hh)[0]) / (2 * hh * S_)
Rr = solG.sol(tt)[0] / S_
print(f" RK45: max |R' - mu(1 + 1/R - R)| = {np.max(np.abs(Rp - mu_ * (1 + 1 / Rr - Rr))):.2e}  (central difference of the dense output, h=1e-6)")
