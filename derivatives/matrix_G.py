import sympy as sp
import numpy as np

u, t11, t21, t31, t12, t22, t32 = sp.symbols(
    "u, t11, t21, t31, t12, t22, t32")

# f1, f2, f3, f4, f5 =sp.symbols("f1, f2, f3, f4, f5")
# df1, df2, df3, df4, df5 =sp.symbols("df1, df2, df3, df4, df5")
# ddf1, ddf2, ddf3, ddf4, ddf5 =sp.symbols("ddf1, ddf2, ddf3, ddf4, ddf5")

# du3 = -df2 * t21 - df4 * t22
# du2 = df2 * t31 + df4 * t32
# du1 = df5 * u

# ddu3 = -ddf2 * t21 - ddf4 * t22
# ddu2 = ddf2 * t31 + ddf4 * t32
# ddu1 = ddf5 * u

# t1 = f1 * t11 + f3 * t12
# t2 = -du3 + 1/2 * du2 * t1 + du1 * du3
# t3 = du2 + 1/2 * du3 * t1 - du1 * du2

# dt1 = df1 * t11 + df3 * t12
# dt2 = -ddu3 + 1/2 * ddu2 * t1 + ddu1 * du3 + du1 * ddu3
# dt3 = ddu2 + 1/2 * ddu3 * t1 - ddu1 * du2 - du1 * ddu2

# pdu1 = [sp.diff(du1, u), sp.diff(du1, t11), sp.diff(du1, t21),
#        sp.diff(du1, t31), sp.diff(du1, t12), sp.diff(du1, t22),
#        sp.diff(du1, t32)]
# print(pdu1)

# pt3 = [sp.diff(t3, u), sp.diff(t3, t11), sp.diff(t3, t21),
#        sp.diff(t3, t31), sp.diff(t3, t12), sp.diff(t3, t22),
#        sp.diff(t3, t32)]
# print(pt3)

# pt2 = [sp.diff(t2, u), sp.diff(t2, t11), sp.diff(t2, t21),
#        sp.diff(t2, t31), sp.diff(t2, t12), sp.diff(t2, t22),
#        sp.diff(t2, t32)]
# print(pt2)

# pdt1 = [sp.diff(dt1, u), sp.diff(dt1, t11), sp.diff(dt1, t21),
#        sp.diff(dt1, t31), sp.diff(dt1, t12), sp.diff(dt1, t22),
#        sp.diff(dt1, t32)]
# print(pdt1)

# pdt2 = [sp.diff(dt2, u), sp.diff(dt2, t11), sp.diff(dt2, t21),
#        sp.diff(dt2, t31), sp.diff(dt2, t12), sp.diff(dt2, t22),
#        sp.diff(dt2, t32)]
# print(pdt2)

# pdt3 = [sp.diff(dt3, u), sp.diff(dt3, t11), sp.diff(dt3, t21),
#        sp.diff(dt3, t31), sp.diff(dt3, t12), sp.diff(dt3, t22),
#        sp.diff(dt3, t32)]
# print(pdt3)

L = sp.symbols("L")
f1 = lambda x: 1 - 3 * (x/L) ** 2 + 2 * (x/L) ** 3
f2 = lambda x: x * (1 - x/L) ** 2
f3 = lambda x: 1 - f1(x)
f4 = lambda x: (x ** 2) * (x/L - 1) / L
f5 = lambda x: x/L

df1 = lambda x: -6*x/L**2 + 6*x**2/L**3
df2 = lambda x: (1 - x/L)**2 - 2*x*(1 - x/L)/L
df3 = lambda x: 6*x/L**2 - 6*x**2/L**3
df4 = lambda x: 2*x*(-1 + x/L)/L + x**2/L**2
df5 = lambda x: 1/L

ddf1 = lambda x: -6/L**2 + 12*x/L**3
ddf2 = lambda x: -4*(1 - x/L)/L + 2*x/L**2
ddf3 = lambda x: 6/L**2 - 12*x/L**3
ddf4 = lambda x: 2*(-1 + x/L)/L + 4*x/L**2
ddf5 = lambda x: 0

pdu1 = lambda x: np.array([[df5(x), 0, 0, 0, 0, 0, 0]])
pt3 = lambda x: np.array([[-df5(x)*(df2(x)*t31 + df4(x)*t32), f1(x)*(-0.5*df2(x)*t21 - 0.5*df4(x)*t22), -0.5*df2(x)*(f1(x)*t11 + f3(x)*t12), -df2(x)*df5(x)*u + df2(x), f3(x)*(-0.5*df2(x)*t21 - 0.5*df4(x)*t22), -0.5*df4(x)*(f1(x)*t11 + f3(x)*t12), -df4(x)*df5(x)*u + df4(x)]])
pt2 = lambda x: np.array([[df5(x)*(-df2(x)*t21 - df4(x)*t22), f1(x)*(0.5*df2(x)*t31 + 0.5*df4(x)*t32), -df2(x)*df5(x)*u + df2(x), 0.5*df2(x)*(f1(x)*t11 + f3(x)*t12), f3(x)*(0.5*df2(x)*t31 + 0.5*df4(x)*t32), -df4(x)*df5(x)*u + df4(x), 0.5*df4(x)*(f1(x)*t11 + f3(x)*t12)]])
pdt1 = lambda x: np.array([[0, df1(x), 0, 0, df3(x), 0, 0]])
pdt2 = lambda x: np.array([[ddf5(x)*(-df2(x)*t21 - df4(x)*t22) + df5(x)*(-ddf2(x)*t21 - ddf4(x)*t22), f1(x)*(0.5*ddf2(x)*t31 + 0.5*ddf4(x)*t32), -ddf2(x)*df5(x)*u + ddf2(x) - ddf5(x)*df2(x)*u, 0.5*ddf2(x)*(f1(x)*t11 + f3(x)*t12), f3(x)*(0.5*ddf2(x)*t31 + 0.5*ddf4(x)*t32), -ddf4(x)*df5(x)*u + ddf4(x) - ddf5(x)*df4(x)*u, 0.5*ddf4(x)*(f1(x)*t11 + f3(x)*t12)]])
pdt3 = lambda x: np.array([[-ddf5(x)*(df2(x)*t31 + df4(x)*t32) - df5(x)*(ddf2(x)*t31 + ddf4(x)*t32), f1(x)*(-0.5*ddf2(x)*t21 - 0.5*ddf4(x)*t22), -0.5*ddf2(x)*(f1(x)*t11 + f3(x)*t12), -ddf2(x)*df5(x)*u + ddf2(x) - ddf5(x)*df2(x)*u, f3(x)*(-0.5*ddf2(x)*t21 - 0.5*ddf4(x)*t22), -0.5*ddf4(x)*(f1(x)*t11 + f3(x)*t12), -ddf4(x)*df5(x)*u + ddf4(x) - ddf5(x)*df4(x)*u]])