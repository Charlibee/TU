import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#ODE
def make_A(eps):
    return np.array([[1, 1], [2/eps, -1/eps]])

def solve_exact(eps, t_end, y0):
    def system(t, y):
        u, v = y
        return [u + v, (2*u - v)/eps]
    sol = solve_ivp(system, [0, t_end], y0, method='Radau', rtol=1e-13, atol=1e-13)
    return sol.y[:, -1]

#Verfahren 
def implicit_euler(A, h, y0, n_steps):
    y = y0.copy()
    B = np.eye(2) - h * A
    for _ in range(n_steps):
        y = np.linalg.solve(B, y)
    return y

def implicit_midpoint(A, h, y0, n_steps):
    y = y0.copy()
    B1 = np.eye(2) - (h/2) * A
    B2 = np.eye(2) + (h/2) * A
    for _ in range(n_steps):
        y = np.linalg.solve(B1, B2 @ y)
    return y

def rk4(A, h, y0, n_steps):
    y = y0.copy()
    for _ in range(n_steps):
        k1 = A @ y
        k2 = A @ (y + h/2 * k1)
        k3 = A @ (y + h/2 * k2)
        k4 = A @ (y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y

#Parameter 
t_end = 0.1
y0 = np.array([1.0, 4.0])
eps_values = [1e-1, 1e-3]

#h=eps**p
powers = [0,1/2,1,2] 

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for ax, eps in zip(axes, eps_values):
    ref = solve_exact(eps, t_end, y0)
    A = make_A(eps)
    
    rad_errs, gau_errs, rk4_errs = [], [], []
    labels = [rf"$\epsilon^{{{p}}}$" for p in powers]
    
    for p in powers:
        h_target = eps**p
        
        #h im Zeitintervall
        if h_target >= t_end:
            n = 1
            h_act = t_end
        else:
            n = int(np.ceil(t_end / h_target))
            h_act = t_end / n
            
        #Berechnungen
        err_rad = np.linalg.norm(implicit_euler(A, h_act, y0, n) - ref)
        err_gau = np.linalg.norm(implicit_midpoint(A, h_act, y0, n) - ref)
        err_rk4 = np.linalg.norm(rk4(A, h_act, y0, n) - ref)
        
        rad_errs.append(err_rad)
        gau_errs.append(err_gau)
        rk4_errs.append(err_rk4)

        #Cap für instabile RK4-Werte zur besseren Darstellung
        #rk4_errs.append(min(err_rk4, 1e5))

    #Plot
    x = np.arange(len(labels))
    width = 0.25
    
    ax.bar(x - width, rad_errs, width, label='Radau', color='blue', edgecolor='black')
    ax.bar(x, gau_errs, width, label='Gauss', color='red', edgecolor='black')
    ax.bar(x + width, rk4_errs, width, label='RK4', color='green', edgecolor='black')
    
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(f"Fehler bei $t=0.1$, $\\varepsilon = {eps}$")
    ax.set_xlabel("Schrittweite")
    ax.set_ylabel("Globaler Fehler")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend()

plt.tight_layout()
plt.show()