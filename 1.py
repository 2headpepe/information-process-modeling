import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Function, diff, sympify, lambdify, sin, cos, exp
from scipy.integrate import odeint

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

def plot_function(f, f_lin, x_vals, u_val, t_val, x0, f_0, sym_x, sym_u, sym_t):
    f_np = lambdify((sym_x, sym_u, sym_t), f, 'numpy')
    f_lin_np = lambdify((sym_x, sym_u, sym_t), f_lin, 'numpy')

    f_vals = f_np(x_vals, u_val, t_val)
    f_lin_vals = f_lin_np(x_vals, u_val, t_val)

    plt.plot(x_vals, f_vals, label='f(x,u,t)', linestyle='-', color='blue')
    plt.plot(x_vals, f_lin_vals, label='linearized f(x,u,t)', linestyle='--', color='red')
    plt.axvline(x0, color='black', linestyle=':')
    plt.axhline(f_0, color='black', linestyle=':')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x,u,t)')
    plt.title('Function and its Linearization')
    plt.grid()
    plt.show()

def ode_system(f_np, x0, t_vals, u_func):
    def system(x, t):
        u = u_func(t)
        return f_np(x, u, t)
    
    x_vals = odeint(system, x0, t_vals)
    return x_vals

def plot_integrated_functions(f_np, f_lin_np, x0, t_vals):
    # u=0
    u0_func = lambda t: 0
    x_orig_u0 = ode_system(f_np, x0, t_vals, u0_func)
    x_lin_u0 = ode_system(f_lin_np, x0, t_vals, u0_func)

    plt.plot(t_vals, x_orig_u0, label='Original f, u=0', linestyle='-', color='blue')
    plt.plot(t_vals, x_lin_u0, label='Linearized f, u=0', linestyle='--', color='blue')

    plt.legend()
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.title('Comparison of Integrated Functions (u=0)')
    plt.grid()
    plt.show()

    # u=0.1*sin(t)
    u1_func = lambda t: 0.1 * np.sin(t)

    x_orig_u1 = ode_system(f_np, x0, t_vals, u1_func)
    x_lin_u1 = ode_system(f_lin_np, x0, t_vals, u1_func)

    plt.plot(t_vals, x_orig_u1, label='Original f, u=0.1*sin(t)', linestyle='-', color='red')
    plt.plot(t_vals, x_lin_u1, label='Linearized f, u=0.1*sin(t)', linestyle='--', color='red')

    plt.legend()
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.title('Comparison of Integrated Functions (u=0.1*sin(t))')
    plt.grid()
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

    f_np = lambdify((x, u, t), f, 'numpy')
    f_lin_np = lambdify((x, u, t), f_lin, 'numpy')
    
    range_size = 1
    x_vals = np.linspace(x0 - range_size, x0 + range_size, 100)
    u_vals = np.linspace(u0 - range_size, u0 + range_size, 100)
    t_vals = np.linspace(0, 10, 100)

    plot_function(f, f_lin, x_vals, u0, t0, x0, f_0, x, u, t)
    plot_integrated_functions(f_np, f_lin_np, x0-range_size, t_vals)

if __name__ == "__main__":
    main()
