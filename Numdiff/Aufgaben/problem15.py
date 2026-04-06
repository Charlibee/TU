import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from butcher_tableaux import *
from methods import *
from visualize import *

#ODE
def f(t, y):
    return -200 * t * y**2

def exact(t):
    return 1 / (1 + 100 * t**2)

t0, T, y0 = 0.0, 1.0, 1.0

#Fehler vs. tau
taus=np.array([10**(-j) for j in range(1,9)])
errors=[]
for tau in taus:
    y_end,ts,ys,hs,nf,nr=richardson_os(f,t0,T,y0,tau,b_rk4,A_rk4,c_rk4)
    errors.append(abs(y_end-exact(T)))
    print(f"tau={tau:.0e}: err={errors[-1]:.2e}, Schritte={len(hs)}, Fevals={nf}, Ablehnungen={nr}")

fig1=plot_error_vs_tau(errors,taus,p=4)
fig1.savefig('error_vs_tau.png',dpi=150); plt.close(fig1)

#Schrittweiten für tau=1e-7
tau7=1e-7
y_end,ts7,ys7,hs7,nf,nr=richardson_os(f,t0,T,y0,tau7,b_rk4,A_rk4,c_rk4)
print(f"\ntau=1e-7: Schritte={len(hs7)}, Fevals={nf}, Ablehnungen={nr}")

fig2=plot_stepsizes(ts7,hs7,tau=tau7)
fig2.savefig('stepsizes.png',dpi=150); plt.close(fig2)