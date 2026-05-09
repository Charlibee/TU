import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor, lu_solve

N = 20
h = 1.0 / N
T = 0.1
x = np.linspace(0, 1, N+1)[1:-1]  

#g(x)
U0 = np.exp(-30 * (x - 0.5)**2)

#M_h
main_diag = -2 * np.ones(N-1)
off_diag = np.ones(N-2)
M_h = (1/h**2) * (np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1))

#expliziter  euler
tau_exp = 0.4 * (h**2) #bisschen kleiner als berechnet
steps_exp = int(T / tau_exp)
U_exp = U0.copy()
norms_exp = [np.max(np.abs(U_exp))]

for _ in range(steps_exp):
    U_exp = U_exp + tau_exp * (M_h @ U_exp)
    norms_exp.append(np.max(np.abs(U_exp)))

#impliziter euler
tau_imp = 0.005 
steps_imp = int(T / tau_imp)
U_imp = U0.copy()
norms_imp = [np.max(np.abs(U_imp))]

A = np.eye(N-1) - tau_imp * M_h
#LU Zerlegung
lu, p = lu_factor(A)

for _ in range(steps_imp):
    U_imp = lu_solve((lu, p), U_imp)
    norms_imp.append(np.max(np.abs(U_imp)))


plt.figure(figsize=(10, 5))

#Plot für Explizit
plt.subplot(1, 2, 1)
plt.plot(norms_exp, color='red')
plt.title(f"Explizit")
plt.xlabel("Schritte")
plt.ylabel("max|U|")

#Plot für Implizit
plt.subplot(1, 2, 2)
plt.plot(norms_imp, color='blue')
plt.title(f"Implizit")
plt.xlabel("Schritte")

plt.tight_layout()
plt.show()

print(f"Stabilitätsgrenze (h^2/2): {0.5*h**2:.6f}")
print(f"Gewähltes tau_exp:         {tau_exp:.6f}")