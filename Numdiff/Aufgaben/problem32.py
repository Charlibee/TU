import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def solve_exact(eps, t_end, y0):
    def system(t, y):
        u, v = y
        return [u + v, (2*u - v)/eps]
    sol = solve_ivp(system, [0, t_end], y0, method='Radau',
                    rtol=1e-12, atol=1e-12)
    return sol.y[:, -1]

#methods
def make_A(eps):
    return np.array([[1, 1], [2/eps, -1/eps]])

def implicit_euler(A, h, y0, n_steps):
    """1-stage Radau-IIA = impliziter Euler"""
    y = y0.copy()
    B = np.eye(2) - h * A
    for _ in range(n_steps):
        y = np.linalg.solve(B, y)
    return y

def implicit_midpoint(A, h, y0, n_steps):
    """1-stage Gauss = implizite Mittelpunktregel"""
    y = y0.copy()
    B1 = np.eye(2) - (h/2) * A
    B2 = np.eye(2) + (h/2) * A
    for _ in range(n_steps):
        y = np.linalg.solve(B1, B2 @ y)
    return y

def rk4(A, h, y0, n_steps):
    """RK4 für lineares System"""
    y = y0.copy()
    B = (np.eye(2) + h*A + (h**2/2)*A@A +
         (h**3/6)*A@A@A + (h**4/24)*A@A@A@A)
    for _ in range(n_steps):
        y = B @ y
    return y

#Parameter
t_end = 0.1
y0    = np.array([1.0, 4.0])
eps_values = [1.0, 1e-1, 1e-2, 1e-3]
h_values   = [t_end / 2**k for k in range(1, 10)] 

#Fehler
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Fehler bei $t=0.1$ vs Schrittweite $h$", fontsize=13)

for ax, eps in zip(axes.flat, eps_values):
    ref = solve_exact(eps, t_end, y0)
    A   = make_A(eps)

    err_rad, err_gau, err_rk4 = [], [], []

    for h in h_values:
        n = max(1, int(round(t_end / h)))
        h_act = t_end / n

        err_rad.append(np.linalg.norm(implicit_euler(A, h_act, y0, n) - ref))
        err_gau.append(np.linalg.norm(implicit_midpoint(A, h_act, y0, n) - ref))

        # RK4: nur wenn Stabilitätsbedingung h < 2.79*eps erfüllt
        rk4_val = rk4(A, h_act, y0, n)
        e = np.linalg.norm(rk4_val - ref)
        err_rk4.append(e if e < 1e10 else np.nan)

    hs = np.array(h_values)
    ax.loglog(hs, err_rad, 'b-o',  label='Radau-IIA (1-stage)', markersize=4)
    ax.loglog(hs, err_gau, 'g-s',  label='Gauss (1-stage)',     markersize=4)
    ax.loglog(hs, err_rk4, 'r-^',  label='RK4',                 markersize=4)

    # Referenzsteigungen
    ax.loglog(hs, 2*hs**1, 'b--', alpha=0.3, label='$O(h)$')
    ax.loglog(hs, 2*hs**2, 'g--', alpha=0.3, label='$O(h^2)$')
    ax.loglog(hs, 2*hs**4, 'r--', alpha=0.3, label='$O(h^4)$')

    ax.set_title(f"$\\epsilon = {eps}$")
    ax.set_xlabel("$h$")
    ax.set_ylabel("$\\|$Fehler$\\|$")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/Users/carlottaniedermaier/TU/Numdiff/error32", dpi=150)
plt.close()

