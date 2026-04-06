import numpy as np
import matplotlib.pyplot as plt

def bs_step(f,t,y,h,b,beta,A,c):
    k0=f(t,y)
    k1=f(t+c[1]*h,y+h*A[1,0]*k0)
    k2=f(t+c[2]*h,y+h*(A[2,0]*k0+A[2,1]*k1))
    k3=f(t+c[3]*h,y+h*(A[3,0]*k0+A[3,1]*k1+A[3,2]*k2))
    k=np.array([k0,k1,k2,k3])
    return b@k,beta@k

def richardson_extrapolate(f,t,y,h,b,beta,A,c,p=2):
    y1=bs_step(f,t,y,h,b,beta,A,c)
    y12=bs_step(f,t,y,h/2,b,beta,A,c)
    y2=bs_step(f,t+h/2,y12,h/2,b,beta,A,c)

    factor = 2**p
    y_extr = (factor * y2 - y1) / (factor - 1)

    return y_extr

def adaptive_rk(f,t0,T,y0,tau,b,beta,A,c,p=2):
    lam,rho=2.0,0.8
    h=min(0.01,tau**(1/p))
    hmin=tau

    t,y=t0,float(y0)
    ts,hs,n_reject=[t],[],0

    while True:
        h=min(T-t,max(hmin,h))
        Ph,P=bs_step(f,t,y,h,b,beta,A,c)

        diff=max(abs(h*(Ph-P)),1e-14)
        H = rho * h * (tau / diff) ** (1/p)
        H=min(H,lam*h)

        if h<=H or h<=hmin:
            t+=h
            y+=h*Ph
            ts.append(t)
            hs.append(h)
            if t>=T:
                break
            h=min(H,lam*h)
        else:
            n_reject+=1
            h=min(H,h/lam)

    return y,np.array(ts),np.array(hs),n_reject,len(hs)



