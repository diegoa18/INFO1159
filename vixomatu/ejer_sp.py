import sympy as sp


def resumen():
    x = sp.Symbol("x")
    f = sp.sympify("x**2 + 3*x + 2")
    df = sp.diff(f, x)
    dfx = sp.lambdify(x, df)
    return dfx(2)


resultado = resumen()
print(resultado)
