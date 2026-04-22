import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from butcher_tableaux import *
from methods import *
from visualize import *

import numpy as np
import matplotlib.pyplot as plt

from butcher_tableaux import A_GL, b_GL, c_GL
from methods import integrate

T = 1.0
tol = 1e-10

# Problems
problems = [
    ("nonlinear", lambda t, y: -200*t*y**2,
     lambda t, y: -400*t*y,
     lambda t: 1/(1 + 100*t**2)),

    ("linear -1", lambda t, y: -y,
     lambda t, y: -1,
     lambda t: np.exp(-t)),

    ("linear -25", lambda t, y: -25*y,
     lambda t, y: -25,
     lambda t: np.exp(-25*t)),
]

N_lst = np.array([10, 20, 40, 80, 160])

results = {}

for name, f, fy, exact in problems:
    errs = []
    max_it = 0

    for N in N_lst:
        yT, it = integrate(f, fy, 0.0, 1.0, T, N, A_GL, b_GL, c_GL, tol)

        errs.append(abs(yT - exact(T)))
        max_it = max(max_it, it)

    results[name] = (np.array(errs), max_it)

#plots

hs = T / N_lst

plt.figure()

for (name, (err, it)) in results.items():
    plt.loglog(hs, err, 'o-', label=f"{name} (it={it})")

plt.loglog(hs, hs**4, '--', label="O(h^4)")

plt.xlabel("h")
plt.ylabel("error at T=1")
plt.grid(True, which="both")
plt.legend()
plt.title("2-stage Gauss IRK convergence")

plt.show()

#output

print("\nConvergence summary")
print("-"*50)

for name, (err, it) in results.items():
    err = np.array(err)
    order = np.log(err[-2]/err[-1]) / np.log(2)
    print(f"{name:15s} | max it={it:2d} | order≈{order:.2f}")