"""Figures for the manuscript (white background, print-safe: colour + line style)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))   # reduced_rd.py, lattice.py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import reduced_rd as M
from lattice import lam_lattice, N2

C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", ink="#0b0b0b", grey="#52514e")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": C["grey"], "axes.labelcolor": C["ink"], "text.color": C["ink"],
                     "xtick.color": C["grey"], "ytick.color": C["grey"], "legend.frameon": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
PHI, R, DL = M.PHI, M.R_THR, M.D_THR
p = M.P

# ---- Fig 1: dispersion -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.6, 3.2))
k = np.linspace(0, 6, 400)
d = p["d"]
ax.axhspan(0, R, color="#eef1f6"); ax.axhspan(R, DL, color="#f6f1ea"); ax.axhspan(DL, 9, color="#f4eeee")
ax.plot(k, p["b"] + d * k**2, color=C["blue"], lw=2, ls="-", label=r"$u$-block")
ax.plot(k, p["g"] + d * k**2, color=C["orange"], lw=2, ls="--", label=r"$v$, $w$ blocks")
ax.axhline(R, color=C["grey"], lw=0.8); ax.axhline(DL, color=C["grey"], lw=0.8)
kU, kV = np.sqrt((DL - p["b"]) / d), np.sqrt((DL - p["g"]) / d)
for kk, c in ((kU, C["blue"]), (kV, C["orange"])):
    ax.plot([kk], [DL], "o", ms=5, color=c, mec="white", mew=1)
ax.text(5.95, 1.3, "visible", ha="right", color=C["grey"]); ax.text(5.95, 3.35, "intermediate", ha="right", color=C["grey"])
ax.text(5.95, 8.4, "singular", ha="right", color=C["grey"])
ax.text(0.05, R + 0.08, r"$R=\varphi^2$", va="bottom", color=C["grey"]); ax.text(0.05, DL + 0.08, r"$\delta=\varphi^3$", va="bottom", color=C["grey"])
ax.annotate(rf"$k={kU:.2f}$", (kU, DL), (kU + 0.35, DL + 1.5), arrowprops=dict(arrowstyle="-", color=C["grey"]), color=C["blue"])
ax.annotate(rf"$k={kV:.2f}$", (kV, DL), (kV + 0.15, DL - 1.55), arrowprops=dict(arrowstyle="-", color=C["grey"]), color=C["orange"])
ax.set_xlim(0, 6); ax.set_ylim(0, 9); ax.set_xlabel(r"wavenumber $k$"); ax.set_ylabel(r"decay rate $\lambda$")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88))
fig.savefig(OUT / "fig_dispersion.pdf"); fig.savefig(OUT / "fig_dispersion.png"); plt.close(fig)

# ---- Fig 2: collapse ---------------------------------------------------------------
u = (np.arange(4000) + 0.5) / 4000
s = np.sin(np.pi * u) ** 2
Tref = np.sort((s[:, None] + s[None, :]).ravel())
xs = np.geomspace(0.01, 5, 300)
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
for ax, name, title in ((axs[0], "v", r"$v$, $w$ blocks"), (axs[1], "u", r"$u$-block")):
    b = {"v": p["g"], "u": p["b"]}[name]; c = DL - b
    Sinf = np.array([1 - np.searchsorted(Tref, c / (4 * x), side="right") / Tref.size for x in xs])
    cols = [C["blue"], C["orange"], C["aqua"], C["yellow"]]
    for N, col, ls in zip((16, 32, 64, 256), cols, ("-", "--", "-.", ":")):
        Lam, _ = lam_lattice(N); L = np.sort(Lam.ravel()) / N**2
        S = np.array([1 - np.searchsorted(L, c / x, side="right") / L.size for x in xs])
        ax.plot(xs, S, color=col, ls=ls, lw=1.6, label=f"$N={N}$")
    ax.plot(xs, Sinf, color=C["ink"], lw=0.9, label=r"$N\to\infty$")
    ax.axvline(c / 4, color=C["grey"], lw=0.7); ax.text(c / 4 * 1.06, 0.04, r"$x_{50}=(\delta-b_j)/4$", color=C["grey"], fontsize=8)
    ax.set_xscale("log"); ax.set_title(title, fontsize=9); ax.set_xlabel(r"$x=dN^2$")
axs[0].set_ylabel("fraction of singular modes"); axs[0].legend(loc="upper left", fontsize=8)
fig.savefig(OUT / "fig_collapse.pdf"); fig.savefig(OUT / "fig_collapse.png"); plt.close(fig)

# ---- Fig 3: nonlinear return (uniform) ------------------------------------------------
x0 = M.fixed_point(); J0 = M.jacobian_k(0.0)
cases = [("$v,w$ (+50%, $-$30%)", x0 * [1, 1.5, 0.7], C["blue"], "-"), ("$u$ (+40%)", x0 * [1.4, 1, 1], C["orange"], "--"),
         ("$u$ ($-$60%)", x0 * [0.4, 1, 1], C["aqua"], "-."), ("all (+40%, +50%, $-$30%)", x0 * [1.4, 1.5, 0.7], C["yellow"], ":")]
fig, ax = plt.subplots(figsize=(4.6, 3.2))
t = np.linspace(0, 8, 400)
for name, xi, col, ls in cases:
    sol = solve_ivp(lambda tt, x: M.local_rhs(x), [0, 8], xi, rtol=1e-13, atol=1e-16, dense_output=True)
    nl = np.array([np.linalg.norm(sol.sol(tt) - x0) for tt in t])
    ax.semilogy(t, nl, color=col, ls=ls, lw=1.8, label=name)
    lin = np.array([np.linalg.norm(expm(J0 * tt) @ (xi - x0)) for tt in t])
    ax.semilogy(t, lin, color=C["grey"], lw=0.6)
ax.set_ylim(1e-9, 3); ax.set_xlabel(r"time $t$"); ax.set_ylabel(r"distance to the fixed point")
ax.legend(loc="upper right", fontsize=7.5)
ax.text(0.2, 2e-9, "grey: linearised solution", fontsize=7.5, color=C["grey"])
fig.savefig(OUT / "fig_nonlinear.pdf"); fig.savefig(OUT / "fig_nonlinear.png"); plt.close(fig)

# ---- Fig 4: adaptive threshold ratio and mode count -----------------------------------
mu = 1.0
f = lambda tt, y: mu * (1 + 1 / y - y)
dd, N = 2e-3, 256
Lam, _ = lam_lattice(N); rates = np.sort((R + dd * Lam).ravel())
count = lambda delta: int(np.searchsorted(rates, delta, side="right") - np.searchsorted(rates, R * (1 + 1e-9), side="right"))
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.9))
tau = np.linspace(0, 8, 801)
for rho0, col, ls in ((0.4, C["blue"], "-"), (1.2, C["orange"], "--"), (3.0, C["aqua"], "-.")):
    sol = solve_ivp(f, [0, 8], [rho0], rtol=1e-12, atol=1e-14, dense_output=True)
    rho = sol.sol(tau)[0]
    axs[0].plot(tau, rho, color=col, ls=ls, lw=1.8, label=rf"$\rho(0)={rho0}$")
    if rho0 >= 1.05:
        axs[1].step(tau, [count(R * r) for r in rho], where="post", color=col, ls=ls, lw=1.8, label=rf"$\rho(0)={rho0}$")
axs[0].axhline(PHI, color=C["grey"], lw=0.8); axs[0].text(7.9, PHI + 0.08, r"$\varphi$", ha="right", color=C["grey"])
axs[0].set_xlabel(r"adaptation time $\tau$"); axs[0].set_ylabel(r"$\rho(\tau)=\delta(\tau)/\lambda_R$"); axs[0].legend(fontsize=8, loc="upper right")
axs[1].axhline(68, color=C["grey"], lw=0.8); axs[1].text(7.9, 84, r"$N_2(t_\delta)-N_2(t_R)=68$", ha="right", fontsize=8, color=C["grey"])
axs[1].set_xlabel(r"adaptation time $\tau$"); axs[1].set_ylabel(r"intermediate modes $I_j(\tau)$"); axs[1].set_ylim(0, 230)
fig.savefig(OUT / "fig_adaptive.pdf"); fig.savefig(OUT / "fig_adaptive.png"); plt.close(fig)
print("figures written")
