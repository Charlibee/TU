import numpy as np
import matplotlib.pyplot as plt

# Expliziten Euler definieren
def explicit(G, tau, h):
    y = [np.array(G)]
    N = len(G)
    M = (np.diag(-2*np.ones(N)) 
        + np.diag(np.ones(N-1), 1) 
        + np.diag(np.ones(N-1), -1)) / h**2
    A = np.eye(N) + tau*M
    
    for l in range(N):
        y.append(A @ y[l])
    
    return y

def implicit(G, tau, h):
    y = [np.array(G)]
    N = len(G)
    M = (np.diag(-2*np.ones(N)) 
        + np.diag(np.ones(N-1), 1) 
        + np.diag(np.ones(N-1), -1)) / h**2
    A = np.eye(N) - tau*M
    
    for l in range(N):
        x = np.linalg.solve(A,y[l])
        y.append(x)
    
    return y



# x und G initialisieren
N_list = [i for i in range(5, 15)]
h_exp = 5
N = 2**(h_exp)      # N-1 ... Länge aller Vektoren
h = 2**(-h_exp)     # da h = 1/N

x = [i*h for i in range(1, N)]
G = [np.exp(-30*(xi - 1/2)**2) for xi in x]



# Plots vorbereiten für expl. Euler
fig, axes = plt.subplots(2, 5, figsize=(18, 7))
axes = axes.flatten()  # damit wir einfach axes[0], axes[1], ... schreiben können

for idx, tau_exp in enumerate(N_list):
    tau = 2**(-tau_exp)
    
    y = explicit(G, tau, h)     # U_n berechnen 

    norms = [np.max(np.abs(y[i])) for i in range(len(y))]   # ||U_n|| für jedes n berechnen
    times = [i*tau for i in range(len(norms)) ]

    # Stabilitätsbedingung überprüfen
    c = min([h**2/(2*np.abs(np.cos(j*np.pi/N)-1)) for j in range(1,N)])  # Konstante abhängig von h definieren
    stable = tau < c
    label = "τ < c(h) (stabil)" if stable else "τ ≥ c(h) (instabil)"

    axes[idx].semilogy(times, norms)
    axes[idx].set_title(f"τ=2^-{tau_exp}\n{label}", fontsize=9)
    axes[idx].set_xlabel("Zeit t")
    axes[idx].set_ylabel("$\\|U_n\\|_\\infty$")
    axes[idx].grid(True)

plt.suptitle(f"Explizites Euler, h=2^-{h_exp}", fontsize=13)
plt.tight_layout()
plt.show()



# Plots vorbereiten für impl. Euler
fig, axes = plt.subplots(2, 5, figsize=(18, 7))
axes = axes.flatten()  # damit wir einfach axes[0], axes[1], ... schreiben können

for idx, tau_exp in enumerate(N_list):
    tau = 2**(-tau_exp)
    
    y = implicit(G, tau, h)     # U_n berechnen 

    norms = [np.max(np.abs(y[i])) for i in range(len(y))]   # ||U_n|| für jedes n berechnen
    times = [i*tau for i in range(len(norms)) ]

    # Stabilitätsbedingung überprüfen
    c = min([h**2/(2*np.abs(np.cos(j*np.pi/N)-1)) for j in range(1,N)])  # Konstante abhängig von h definieren
    stable = tau < c
    label = "τ < c(h) (stabil)" if stable else "τ ≥ c(h) (instabil)"

    axes[idx].semilogy(times, norms)
    axes[idx].set_title(f"τ=2^-{tau_exp}\n{label}", fontsize=9)
    axes[idx].set_xlabel("Zeit t")
    axes[idx].set_ylabel("$\\|U_n\\|_\\infty$")
    axes[idx].grid(True)

plt.suptitle(f"Implizites Euler, h=2^-{h_exp}", fontsize=13)
plt.tight_layout()
plt.show()