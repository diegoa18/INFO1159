import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

x = sp.symbols("x")
try:
    func = sp.lambdify(x, sp.sympify(input("ingrese f(x): ")), "numpy")
except:
    print("función invalida")
    exit()

xa, xb = float(input("ingrese xa: ")), float(input("ingrese xb: "))
if xa > xb:
    xa, xb = xb, xa

dl = float(input("ingrese Δλ (0 < Δλ ≤ 1): "))
if not 0 < dl <= 1:
    print("Δλ debe estar en (0, 1]")
    exit()

lda = np.arange(0, 1 + dl, dl)
is_concave = np.all(
    func(lda * xa + (1 - lda) * xb) >= lda * func(xa) + (1 - lda) * func(xb)
)
is_convex = np.all(
    func(lda * xa + (1 - lda) * xb) <= lda * func(xa) + (1 - lda) * func(xb)
)

if is_concave and not is_convex:
    print("cóncava")
elif is_convex and not is_concave:
    print("convexa")
elif is_convex and is_concave:
    print("lineal")
else:
    print("no es ni concava ni convexa")

t = np.linspace(xa - 1, xb + 1, 500)
plt.plot(t, func(t), "b-", label="f(x)")
plt.plot([xa, xb], [func(xa), func(xb)], "r--", label="recta")
plt.show()
