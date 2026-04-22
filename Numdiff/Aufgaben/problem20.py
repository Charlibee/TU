import sympy as sp

def generate_butcher_tableau(N, method_type='closed'):
    tau = sp.symbols('tau')
    
    #knoten nach aufgabe
    if method_type == 'closed':
        c = [sp.Rational(l, N) for l in range(N + 1)]
    else: # open
        c = [sp.Rational(l + 1, N + 2) for l in range(N + 1)]
    
    s = len(c)
    
    #Ls
    def lagrange_poly(j, nodes):
        poly = 1
        for m in range(len(nodes)):
            if m != j:
                poly *= (tau - nodes[m]) / (nodes[j] - nodes[m])
        return sp.simplify(poly)

    L = [lagrange_poly(j, c) for j in range(s)]
    
    #integrierte Koeffizienten
    b = [sp.integrate(L[j], (tau, 0, 1)) for j in range(s)]
    A = [[sp.integrate(L[j], (tau, 0, c[i])) for j in range(s)] for i in range(s)]
    
    print("Knoten:", c)
    print("Gewichte:", b)
    print("Matrix:")
    for row in A:
        print(row)


generate_butcher_tableau(1, 'closed')
#generate_butcher_tableau(1, 'open')