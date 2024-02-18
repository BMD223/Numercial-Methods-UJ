import numpy as np
import matplotlib.pyplot as plt

def function1(x):
    return 1 / (1 + 50 * x**2)

def function2(x):
    return np.cos(x**2)

def function3(x):
    return 1 / (1 + 24*x**2/np.e)

def nodes1(n):
    return np.linspace(-1, 1, n + 1)

def nodes2(n):
    return np.cos((2 * np.arange(n + 1) + 1) * np.pi / (2 * (n + 1)))

def lagrange(x, nodes, f):
    result = 0
    n = len(nodes)
    for i in range(n):
        temp = 1
        for j in range(n):
            if i != j:
                temp *= (x - nodes[j]) / (nodes[i] - nodes[j])
        result += f(nodes[i]) * temp
    return result

def plot_interpolating_polynomials(nodes_func, function, n_values, func_name):
    x_values = np.linspace(-1.0, 1.0, 1000)
    plt.figure(figsize=(12, 6))
    plt.title(f"Interpolacja dla: {func_name}")

    for n in n_values:
        nodes = nodes_func(n)
        y_interpolated = [lagrange(x, nodes, function) for x in x_values]
        plt.plot(x_values, y_interpolated, label=f'n={n}')

    y_original = [function(x) for x in x_values]
    plt.plot(x_values, y_original, label='Oryginalna funkcja', color='black', linewidth=2)
    # if(function==function3):
    #     plt.ylim(-100, 50)
    plt.legend()
    plt.show()

def plot_interpolating_polynomials_modified(nodes_func, function, n_values, func_name):
    x_values = np.linspace(-1.0, 1.0, 1000)
    plt.figure(figsize=(12, 6))
    plt.title(f"Interpolacja dla: {func_name}")

    for n in n_values:
        if function == function2 and n > 3:  # Limit na f2 do 3
            continue

        nodes = nodes_func(n)
        y_interpolated = [lagrange(x, nodes, function) for x in x_values]
        plt.plot(x_values, y_interpolated, label=f'n={n}')

    y_original = [function(x) for x in x_values]
    plt.plot(x_values, y_original, label='Oryginalna funkcja', color='black', linewidth=2)
    
    if function == function3:
        plt.ylim(-100, 50)  
    
    plt.legend()
    plt.show()


def main():
    n_values = [2, 3, 6, 10, 12]
    plot_interpolating_polynomials(nodes1, function1, n_values, "Funkcja 1: 1/(1+50*x^2)")
    plot_interpolating_polynomials(nodes2, function1, n_values, "Funkcja 1: 1/(1+50*x^2) z wezlami 2")
    plot_interpolating_polynomials_modified(nodes1, function2, n_values, "Funkcja 2: cos(x^2)")
    plot_interpolating_polynomials_modified(nodes2, function2, n_values, "Funkcja 2: cos(x^2) z wezlami 2")
    plot_interpolating_polynomials(nodes1, function3, n_values, "Funkcja 3: 1/(24/e *x^2 +1)")
    plot_interpolating_polynomials(nodes2, function3, n_values, "Funkcja 3: 1/(24/e *x^2 +1) z wezlami 2")

main()
