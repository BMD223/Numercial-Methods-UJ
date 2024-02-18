import numpy as np
import matplotlib.pyplot as plt

def F(x, a, b, c, d):
    return a * x**2 + b * np.sin(x) + c * np.cos(5*x) + d * np.exp(-x)

def G(x, a, b, c, d):
    return a * np.log(x) + b * 1/(x+np.e)  + c * np.sin(x) + d * np.cos

def load_data(filename):
    return np.loadtxt(filename, delimiter=',')

def calculate_coefficients(X, y):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    X_inv = Vt.T @ np.linalg.inv(np.diag(S)) @ U.T
    return X_inv @ y

def plot_approximation(x, y, x_plot, y_plot, function_label, title):
    plt.scatter(x, y, label='Punkty')
    plt.plot(x_plot, y_plot, label=function_label, color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(title)
    plt.show()

def open_and_process_data(filename):
    points = load_data(filename)
    x = points[:, 0]
    y = points[:, 1]
    return x, y

def generate_noisy_data(num_points, noise_mean, noise_std):
    x = np.linspace(0, 10, num_points)
    delta_y = np.random.normal(noise_mean, noise_std, len(x))
    y_val = G(x, 0.5, 12.0, 15.0, 18.0)
    y_noisy = y_val + delta_y
    return x, y_noisy

def plot_original_and_approximated(x, y_noisy, x_plot, y_plot, original_function_label, approximated_function_label, title):
    plt.scatter(x, y_noisy, label='Noisy Data')
    plt.plot(x_plot, y_plot, label=approximated_function_label, color='red')
    plt.plot(x, G(x, 0.5, 12.0, 15.0, 18.0), label=original_function_label, linestyle='--', color='green')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(title)
    plt.show()

def main():
    x, y = open_and_process_data('Zadanie 8/Z8a/dane.txt')

    X = np.column_stack([x**2, np.sin(x), np.cos(5*x), np.exp(-x)])
    coefficients = calculate_coefficients(X, y)
    print("Optimal coefficients for function F(x):")
    print(f'a: {coefficients[0]}, b: {coefficients[1]}, c: {coefficients[2]}, d: {coefficients[3]}')
    print()

    x_plot = np.linspace(min(x), max(x), 1000)
    y_plot = F(x_plot, *coefficients)
    plot_approximation(x, y, x_plot, y_plot, f'F(x) = {coefficients[0]}*$x^2$ + {coefficients[1]}*sin(x) + {coefficients[2]}*cos(5x) + {coefficients[3]}*exp(-x)', 'Approximation for subpoint a')

    np.random.seed(58)
    for num_points, noise_std, noise_mean in [(60, 3, 0.7), (150, 20, 10), (350, 3, 0.1)]:
        x, y_noisy = generate_noisy_data(num_points, noise_mean, noise_std)
        X = np.column_stack([x**2, np.sin(5*x), np.cos(2*x), np.exp(-5*x)])
        coefficients = calculate_coefficients(X, y_noisy)
        print(f"Optimal coefficients for function G(x) (for {num_points} points with noise mean {noise_mean} and standard deviation {noise_std}):")
        print(f'a: {coefficients[0]}, b: {coefficients[1]}, c: {coefficients[2]}, d: {coefficients[3]}')
        x_plot = np.linspace(min(x), max(x), 1000)
        y_plot = G(x_plot, *coefficients)
        plot_original_and_approximated(x, y_noisy, x_plot, y_plot, 'Original function', f'G(x) = {coefficients[0]}*$x^3$ + {coefficients[1]}*sin(2x) + {coefficients[2]}*cos(6x) + {coefficients[3]}*exp(-2x)', f'Approximation for G(x) with {num_points} points and noise mean {noise_mean} and std deviation {noise_std}')

if __name__ == "__main__":
    main()
