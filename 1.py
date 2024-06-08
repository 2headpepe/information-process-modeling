import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Function, diff, sympify, sin, cos, exp

# Примеры функций
PREDEFINED_FUNCTIONS = {
    "1": "x**2 * u + u**2 * t - x * t",
    "2": "sin(x) * cos(u) + exp(t) - x * u",
    "3": "x**3 + u**3 + t**2 - x * u * t",
    "4": "x**2 + u**2 + t**2",
}

def choose_function():
    print("Choose a function to linearize:")
    for key, func in PREDEFINED_FUNCTIONS.items():
        print(f"{key}: {func}")
    choice = input("Enter the number of the function: ")
    return sympify(PREDEFINED_FUNCTIONS.get(choice, PREDEFINED_FUNCTIONS["1"]))

def compute_derivatives(f, x, u, t):
    df_dx = diff(f, x)
    df_du = diff(f, u)
    df_dt = diff(f, t)
    return df_dx, df_du, df_dt

def evaluate_at_point(f, df_dx, df_du, df_dt, x, u, t, x0, u0, t0):
    f_0 = f.subs({x: x0, u: u0, t: t0})
    df_dx_0 = df_dx.subs({x: x0, u: u0, t: t0})
    df_du_0 = df_du.subs({x: x0, u: u0, t: t0})
    df_dt_0 = df_dt.subs({x: x0, u: u0, t: t0})
    return f_0, df_dx_0, df_du_0, df_dt_0

def linearize_function(f_0, df_dx_0, df_du_0, df_dt_0, x, u, t, x0, u0, t0):
    return f_0 + (x - x0) * df_dx_0 + (u - u0) * df_du_0 + (t - t0) * df_dt_0

def plot_3d_function(f, f_lin, x_vals, u_vals, t_val):
    f_np = np.vectorize(lambda x, u, t: f.subs({'x': x, 'u': u, 't': t}).evalf())
    f_lin_np = np.vectorize(lambda x, u, t: f_lin.subs({'x': x, 'u': u, 't': t}).evalf())
    
    X, U = np.meshgrid(x_vals, u_vals)
    f_vals = f_np(X, U, t_val)
    f_lin_vals = f_lin_np(X, U, t_val)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_surface(X, U, f_vals, cmap='viridis', alpha=0.7)
    
    ax.plot_surface(X, U, f_lin_vals, cmap='inferno', alpha=0.7)

    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_zlabel('f(x,u,t)')
    ax.set_title('3D Plot of Function and its Linearization')
    plt.show()

def main():
    x, u, t = symbols('x u t')

    f = choose_function()

    df_dx, df_du, df_dt = compute_derivatives(f, x, u, t)

    x0 = float(input("Enter the linearization point for x: "))
    u0 = float(input("Enter the linearization point for u: "))
    t0 = float(input("Enter the linearization point for t: "))

    f_0, df_dx_0, df_du_0, df_dt_0 = evaluate_at_point(f, df_dx, df_du, df_dt, x, u, t, x0, u0, t0)

    f_lin = linearize_function(f_0, df_dx_0, df_du_0, df_dt_0, x, u, t, x0, u0, t0)

    range_size = 1
    x_vals = np.linspace(x0 - range_size, x0 + range_size, 100)
    u_vals = np.linspace(u0 - range_size, u0 + range_size, 100)
    t_vals = np.linspace(t0 - range_size, t0 + range_size, 100)

    plot_3d_function(f, f_lin, x_vals, u_vals, t0)

if __name__ == "__main__":
    main()
