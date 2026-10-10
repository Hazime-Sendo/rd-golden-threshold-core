import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))   # reduced_rd.py, lattice.py
import numpy as np
from scipy.integrate import solve_ivp
import reduced_rd as M
from scipy.optimize import curve_fit

def fit_rate(ts, dd):
    """fit dist ~ sqrt(A + B t + C t^2) exp(-lam t)  (norm of a Jordan-block solution; g = h)."""
    f = lambda t, lam, A, B, C: 0.5 * np.log(np.abs(A + B * t + C * t**2) + 1e-300) - lam * t
    a0 = dd[0] * np.exp(2.6 * ts[0])
    popt, _ = curve_fit(f, ts, np.log(dd), p0=[2.6, a0**2, 0.0, 0.0], maxfev=200000)
    return popt[0]

PHI, R, DL = M.PHI, M.R_THR, M.D_THR
p = M.P
x0 = M.fixed_point()
d = p["d"]
rates0 = np.array(sorted([p["b"], p["g"], p["h"]]))

# ---- 1. numeric J(k) vs analytic  lambda = b_i + d k^2 -------------------
print("== 1. J(k): numeric vs analytic ==")
err = 0
for k in np.linspace(0, 12, 49):
    err = max(err, np.max(np.abs(M.decay_rates(k) - (rates0 + d * k**2))))
print(f"max |numeric - analytic| over k in [0,12]: {err:.2e}")
J0 = M.jacobian_k(0.0)
print("J(0) upper triangle max |.| =", np.max(np.abs(np.triu(J0, 1))))

# ---- 2. dispersion / threshold crossings ---------------------------------
print("\n== 2. classification and crossings ==")
for k in (0.0, 1.0):
    lam = M.decay_rates(k)
    print(f"k={k}: ", np.round(lam, 6), [M.classify(x) for x in lam])
for name, b in (("u", p["b"]), ("v", p["g"]), ("w", p["h"])):
    kR = np.sqrt(max(R - b, 0) / d)
    kD = np.sqrt((DL - b) / d)
    print(f"block {name}: b={b:.6f}  leaves 'visible' at k={kR:.4f}, crosses delta at k={kD:.4f}")

# ---- 3. nonlinear, spatially uniform ------------------------------------
print("\n== 3. nonlinear return (uniform): nonlinear vs linearized trajectory ==")
from scipy.linalg import expm
J0 = M.jacobian_k(0.0)
def dist(x): return np.linalg.norm(x - x0)
cases = {
    "v,w large (+50%,-30%)": x0 * np.array([1, 1.5, 0.7]),
    "u large (+40%)": x0 * np.array([1.4, 1, 1]),
    "all (+40%,+50%,-30%)": x0 * np.array([1.4, 1.5, 0.7]),
    "u down (-60%)": x0 * np.array([0.4, 1, 1]),
}
tchk = (2.0, 4.0, 6.0, 8.0)
for name, xi in cases.items():
    sol = solve_ivp(lambda t, x: M.local_rhs(x), [0, 9], xi, rtol=1e-13, atol=1e-16, dense_output=True)
    ratios = [dist(sol.sol(t)) / np.linalg.norm(expm(J0 * t) @ (xi - x0)) for t in tchk]
    drift = (ratios[-1] - ratios[-2]) / ratios[-1]
    print(f"{name:26s} init dist {dist(xi):.4f}  nonlinear/linear at t={tchk}: "
          + " ".join(f"{r:.5f}" for r in ratios) + f"   (last-step drift {drift:+.1e})")

# ---- 4. basin along u ------------------------------------------------------
print("\n== 4. basin lower boundary in u ==")
u2 = M.u2()
print(f"u_2 = {u2:.6f}  ->  f_crit = u_2/u* = {u2/p['u_star']:.6f}")
def diverges(fac):
    xi = x0.copy(); xi[0] = fac * x0[0]
    ev = lambda t, x: x[0] + 1e3
    ev.terminal = True
    sol = solve_ivp(lambda t, x: M.local_rhs(x), [0, 200], xi, rtol=1e-10, atol=1e-12, events=ev)
    return sol.t_events[0].size > 0
lo, hi = 0.0, 1.0
for _ in range(50):
    mid = 0.5 * (lo + hi)
    if diverges(mid): lo = mid
    else: hi = mid
print(f"bisection boundary f = {hi:.6f}")
print("upper side: f = 10, 100, 1000 diverge?", [diverges(f) for f in (10, 100, 1000)])

# ---- 5. nonlinear, spatially varying (2D periodic, FD) ---------------------
print("\n== 5. nonlinear 2D periodic PDE, spatial perturbation ==")
N = 32; hgrid = 1.0 / N
kap = 0.01
pp = dict(p); pp["d"] = kap
X, Y = np.meshgrid(np.arange(N) * hgrid, np.arange(N) * hgrid, indexing="ij")
def lap(F):
    return (np.roll(F, 1, 0) + np.roll(F, -1, 0) + np.roll(F, 1, 1) + np.roll(F, -1, 1) - 4 * F) / hgrid**2
def rhs(t, y):
    u, v, w = y.reshape(3, N, N)
    return np.concatenate([
        (M.f(u, pp) + kap * lap(u)).ravel(),
        (-pp["g"] * v + pp["a0"] + pp["a1"] * u + kap * lap(v)).ravel(),
        (-pp["h"] * w + pp["c1"] * u + pp["c2"] * v + kap * lap(w)).ravel(),
    ])
lam_fd = 4 * N**2 * np.sin(np.pi / N)**2          # discrete eigenvalue of mode (1,0)
mode = np.cos(2 * np.pi * X)
base = np.concatenate([np.full(N * N, c) for c in x0])
def dev_norm(y): return np.linalg.norm(y - base) * hgrid
def pde_case(name, amp, tmax=6.0):
    dev0 = np.concatenate([amp[0] * mode.ravel() * x0[0], amp[1] * mode.ravel() * x0[1], amp[2] * mode.ravel() * x0[2]])
    y0 = base + dev0
    ts = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    sol = solve_ivp(rhs, [0, tmax], y0, t_eval=ts, rtol=1e-12, atol=1e-15, method="RK45")
    A = M.jacobian_k(0.0) - kap * lam_fd * np.eye(3)
    a0 = np.array(amp) * x0
    out = []
    for i, t in enumerate(ts):
        nl = dev_norm(sol.y[:, i])
        lin = np.linalg.norm(expm(A * t) @ a0) * np.linalg.norm(mode) * hgrid
        out.append(nl / lin)
    print(f"{name:26s} init dev {dev_norm(y0):.4f}  nonlinear/linear(mode (1,0)) at t=2..6: "
          + " ".join(f"{r:.4f}" for r in out))
    # spatial-mean (k=0) component: compare with the uniform linear flow started at t0=3
    Y = sol.y.reshape(3, N, N, -1)
    m = np.array([[Y[c, :, :, i].mean() - x0[c] for c in range(3)] for i in range(len(ts))])
    J0 = M.jacobian_k(0.0)
    t0i = 1
    if np.linalg.norm(m[t0i]) < 1e-10:      # mean component is pure round-off (e.g. w-only perturbation)
        return
    for i in range(t0i + 1, len(ts)):
        pred = expm(J0 * (ts[i] - ts[t0i])) @ m[t0i]
        print(f"      mean component t={ts[i]:.0f}: |m|={np.linalg.norm(m[i]):.3e}, |m|/|linear flow from t=3|={np.linalg.norm(m[i])/np.linalg.norm(pred):.4f}")
print(f"linear decay of mode (1,0): rates {np.round(np.sort(rates0 + kap*lam_fd),5)}; k=0 slowest {rates0[0]:.5f}")
pde_case("small", (0.001, 0.001, 0.001))
pde_case("large (u,v,w +-30..50%)", (0.4, 0.5, -0.3))
pde_case("large, w only", (0.0, 0.0, 0.5))
