"""Three-component reaction-diffusion system with block lower-triangular coupling.

    u_t = d Lap u + f(u)
    v_t = d Lap v - g v + a0 + a1 u
    w_t = d Lap w - h w + c1 u + c2 v

with f(u) = -3 (u - u_star)(u - u_2), u_2 = u_star - b/3, so that
f'(u_star) = -b (b = decay rate of the u-block).  No application-specific
names are used.  Thresholds: R = phi^2, delta = phi^3 (= phi * R).
"""
import numpy as np

PHI = (1 + np.sqrt(5)) / 2
R_THR = PHI**2
D_THR = PHI**3
TOL = 1e-9

P = dict(
    d=0.1,                    # common diffusion coefficient
    u_star=PHI,               # fixed point of the u-block
    b=20 - 10 * PHI,          # u-block decay rate  -f'(u_star) = 3.8197
    g=PHI**2,                 # v-block decay rate
    h=PHI**2,                 # w-block decay rate
    a0=0.06, a1=0.08,         # v forcing
    c1=0.5, c2=0.3,           # w forcing
)


def u2(p=P):
    return p["u_star"] - p["b"] / 3.0


def f(u, p=P):
    return -3.0 * (u - p["u_star"]) * (u - u2(p))


def fp(u, p=P):
    return -3.0 * ((u - p["u_star"]) + (u - u2(p)))


def local_rhs(x, p=P):
    u, v, w = x
    return np.array([
        f(u, p),
        -p["g"] * v + p["a0"] + p["a1"] * u,
        -p["h"] * w + p["c1"] * u + p["c2"] * v,
    ])


def fixed_point(p=P):
    u = p["u_star"]
    v = (p["a0"] + p["a1"] * u) / p["g"]
    w = (p["c1"] * u + p["c2"] * v) / p["h"]
    return np.array([u, v, w])


def jacobian_k(k, p=P, h_fd=1e-6):
    """J(k) = J(0) - k^2 D, with J(0) by central differences at the fixed point."""
    x0 = fixed_point(p)
    J = np.zeros((3, 3))
    for j in range(3):
        e = np.zeros(3)
        e[j] = h_fd
        J[:, j] = (local_rhs(x0 + e, p) - local_rhs(x0 - e, p)) / (2 * h_fd)
    return J - k**2 * p["d"] * np.eye(3)


def decay_rates(k, p=P):
    ev = np.linalg.eigvals(jacobian_k(k, p))
    return np.sort(-ev.real)


def classify(lam, R=R_THR, delta=D_THR, tol=TOL):
    if lam <= R * (1 + tol):
        return "visible"
    if lam <= delta * (1 + tol):
        return "intermediate"
    return "singular"


if __name__ == "__main__":
    print("fixed point", fixed_point())
    for k in (0.0, 1.0):
        lam = decay_rates(k)
        print(f"k={k}: ", np.round(lam, 6), [classify(x) for x in lam])
