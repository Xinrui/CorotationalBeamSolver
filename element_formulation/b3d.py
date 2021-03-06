import sympy as sp

u, t11, t12, t21, t22, t31, t32, x = sp.symbols(
    "u, t11, t12, t21, t22, t31, t32, x")

L, A, I22, I33, Irr, E, G = sp.symbols(
    "L, A, I22, I33, Irr, E, G")

Io = I22 + I33

f1 = 1 - 3 * (x/L) ** 2 + 2 * (x/L) ** 3
f2 = x * (1 - x / L) ** 2
f3 = 1 - f1
f4 = (x ** 2) * (x / L - 1) / L
f5 = 1 - x / L
f6 = 1 - f5

u2 = f2 * t31 + f4 * t32
du2 = sp.diff(u2, x)
ddu2 = sp.diff(du2, x)
u3 = -f2 * t21 - f4 * t22
du3 = sp.diff(u3, x)
ddu3 = sp.diff(du3, x)

t1 = f5 * t11 + f6 * t12
dt1 = sp.diff(t1, x)

k1 = dt1
k2 = -ddu2
k3 = -ddu3

eav = u/L + 1/2/L * \
    sp.integrate(du2 ** 2 + du3 ** 2 + Io / A * dt1 ** 2, (x, 0, L))

phi1 = A * eav ** 2 + I22 * k2 ** 2 + I33 * \
    k3 ** 2 + 1 / 4 * (Irr - Io ** 2 / A) * dt1 ** 4

phi = 1/2 * sp.integrate(E * phi1 + G * Io * k1 ** 2, (x, 0, L))


def gradient(f, v): return sp.Matrix([f]).jacobian(v)


fl = gradient(phi, [u, t11, t21, t31, t12, t22, t32])
kl = sp.hessian(phi, [u, t11, t21, t31, t12, t22, t32])

fl = sp.simplify(fl)
kl = sp.simplify(kl)

print("The local force of Bernoulli element is: ")
print(fl)
print("The local stiffness matrix of Bernoulli element is: ")
print(kl)
