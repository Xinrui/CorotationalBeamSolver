import sympy as sp

u1, w1, t1, u2, w2, t2 = sp.symbols("u1, w1, t1, u2, w2, t2")
X1, Z1, X2, Z2 = sp.symbols("X1, Z1, X2, Z2")

l = sp.sqrt((X2 + u2 - X1 - u1) ** 2 + (Z2 + w2 - Z1 - w1) ** 2)
L = sp.sqrt((X2 - X1) ** 2 + (Z2 - Z1) ** 2)

beta0 = sp.atan((Z2 - Z1)/(X2 - X1))
beta = sp.atan((Z2 + w2 - Z1 - w1)/(X2 + u2 - X1 - u1))

u_bar = l - L
t1_bar = t1 + beta0 - beta
t2_bar = t2 + beta0 - beta

def gradient(f, v): return sp.Matrix([f]).jacobian(v)

pl = [u_bar, t1_bar, t2_bar]
pg = [u1, w1, t1, u2, w2, t2]

B = gradient(pl, pg)
print(sp.latex(sp.simplify(B.T)))