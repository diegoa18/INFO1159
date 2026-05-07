import numpy as np
import sympy as sp

theta = 1e-4
step = 0.3

x, y = sp.symbols("x y")
obj_expression = 1.2 * x + 1.16 * y - theta * (2 * x**2 + y**2 + (x + y) ** 2)
obj = sp.lambdify((x, y), obj_expression, "numpy")

x_grid = np.arange(0, 5001, step)
y_grid = np.arange(0, 5001, step)

X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")

mask = (X >= 0) & (Y >= 0) & (X + Y <= 5000)
X_valid, Y_valid = X[mask], Y[mask]

if X_valid.size > 0:
    factible_vals = obj(X_valid, Y_valid)
    opt_index = np.argmax(factible_vals)
    x_opt, y_opt = X_valid[opt_index], Y_valid[opt_index]
    f_opt = factible_vals[opt_index]


print(f"indice optimo encontrado con step={step}: {x_opt},{y_opt}")
print(f"valor maximo: {f_opt}")
