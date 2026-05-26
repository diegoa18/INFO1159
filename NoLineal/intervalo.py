import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

x = sp.symbols("x")
func = sp.lambdify(x, sp.sympify(input("ingrese f(x): ")), "numpy")
xa, xb = float(input("ingrese xa: ")), float(input("ingrese xb: "))
dl = float(input("ingrese Δλ (0 < Δλ ≤ 1): "))

if xa > xb:
    xa, xb = xb, xa

lda = np.linspace(0, 1, int(1 / dl) + 1)
concava = np.all(
    func(lda * xa + (1 - lda) * xb) >= lda * func(xa) + (1 - lda) * func(xb)
)
convexa = np.all(
    func(lda * xa + (1 - lda) * xb) <= lda * func(xa) + (1 - lda) * func(xb)
)

if concava and not convexa:
    print("cóncava")
elif convexa and not concava:
    print("convexa")
elif convexa and concava:
    print("lineal")
else:
    print("no es ni concava ni convexa")

padding = max(0.2 * (xb - xa), 0.5)
puntos = min(max(200, int(200 * (xb - xa))), 5000)
full = np.linspace(xa - padding, xb + padding, puntos)
interval = np.linspace(xa, xb, puntos)

plt.plot(full, func(full), "k-", label="f(x)")
plt.plot(interval, func(interval), "b-", label="f(λxa + (1 - λ)xb)")
plt.plot([xa, xb], [func(xa), func(xb)], "r--", label="λf(xa) + (1 - λ)f(xb)")
plt.legend(fontsize="large")
plt.show()
