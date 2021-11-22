import sympy as sp

u, t11, t12, t21, t22, t31, t32 = sp.symbols("u, t11, t12, t21, t22, t31, t32")

L, A, I22, I33, Irr, E, G = sp.symbols("L, A, I22, I33, Irr, E, G")

Io = I22 + I33

du = u / L

t1 = (t11 + t12) / 2
t2 = (t21 + t22) / 2
t3 = (t31 + t32) / 2

dt1 = (t12 - t11) / L
dt2 = (t22 - t21) / L
dt3 = (t32 - t31) / L

g12 = -t3
g13 = t2

k1 = dt1
k2 = -dt3
k3 = dt2

phi1 = A * du ** 2 + I22 * k2 ** 2 + I33 * \
    k3 ** 2 + Irr / 4 * dt1 ** 4 * Io * du * dt1 ** 2
phi2 = A * (g12 ** 2 + g13 ** 2) + Io * k1 ** 2
phi = 1/2 * L * (E * phi1 + G * phi2)


def gradient(f, v): return sp.Matrix([f]).jacobian(v)


fl = gradient(phi, [u, t11, t21, t31, t12, t22, t32])
kl = sp.hessian(phi, [u, t11, t21, t31, t12, t22, t32])

fl = sp.simplify(fl)
kl = sp.simplify(kl)

print("The local force of Bernoulli element is: ")
print(fl)
print("The local stiffness matrix of Bernoulli element is: ")
print(kl)
