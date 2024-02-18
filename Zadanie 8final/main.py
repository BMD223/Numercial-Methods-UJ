from typing import List, Callable
import matplotlib.pyplot as plt
import numpy as np
import random
import csv


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class FunctionApproximator:
    # Components of function F
    F_COMPONENTS = [
        lambda x: x ** 2,
        np.sin,
        lambda x: np.cos(5 * x),
        lambda x: np.exp(-x)
    ]

    # Components of function G
    G_COMPONENTS = [
        lambda x: np.log(x) if x > 0 else 0,
        lambda x: 1/(x+np.e) if x != 0 else 1,
        np.sin,
        np.cos
    ]

    G_POINT_COUNTS = [120,240]
    G_EXACT_PARAMS = [7, 5, -1,6]
    G_MAX_DISTURBANCE = 25


    def __init__(self):
        self.graph_density = 50
        self.min_x, self.max_x = None, None
        self.points = self.load_csv('Zadanie 8/dane.csv')
        self.solve_and_graph('F(x)', self.F_COMPONENTS)
        self.optimal_coefficients=[]

        self.min_x, self.max_x = 0.1, 50
        for count in self.G_POINT_COUNTS:
            self.points = self.generate_points(count, self.G_COMPONENTS, self.G_EXACT_PARAMS, self.G_MAX_DISTURBANCE)
            self.solve_and_graph(f'G(x) for {count} points', self.G_COMPONENTS)
            parameters = self.solve_for_params(self.generate_matrix_a(self.G_COMPONENTS))
            self.optimal_coefficients.append(parameters)

    def generate_points(self, count: int, components: List[Callable], exact_params: List[float], max_disturbance: float) -> List[Point]:
        random.seed(10)
        points = [self.create_noisy_point(x, components, exact_params, max_disturbance) for x in np.linspace(self.min_x, self.max_x, count)]
        return points

    def create_noisy_point(self, x: float, components: List[Callable], params: List[float], max_disturbance: float) -> Point:
        y_exact = self.calculate_function(x, params, components)
        y_noisy = y_exact + random.uniform(-1, 1) * max_disturbance
        return Point(x, y_noisy)

    def solve_and_graph(self, title: str, components: List[Callable]):
        matrix_a = self.generate_matrix_a(components)
        parameters = self.solve_for_params(matrix_a)
        self.graph(title, components, parameters)

    def generate_matrix_a(self, components: List[Callable]) -> np.ndarray:
        matrix_a = np.array([[component(point.x) for component in components] for point in self.points])
        return matrix_a

    def solve_for_params(self, matrix_a: np.ndarray) -> np.ndarray:
        at_times_a = np.matmul(matrix_a.T, matrix_a)
        if np.linalg.det(at_times_a) == 0:
            raise ValueError("A.T * A has a determinant of zero. No solution can be found.")

        y_array = np.array([point.y for point in self.points])
        at_times_y = np.matmul(matrix_a.T, y_array)
        return np.linalg.solve(at_times_a, at_times_y)

    def graph(self, title: str, components: List[Callable], parameters: np.ndarray) -> None:
        plt.close()
        self.plot_data_points()
        self.plot_approximation(components, parameters)
        plt.title(title)
        plt.xlabel('x')  # Adding labels for clarity
        plt.ylabel('y')
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_data_points(self):
        x_values, y_values = zip(*[(point.x, point.y) for point in self.points])
        plt.scatter(x_values, y_values, label="Data Points", marker="o", s=14, color='blue')  # Bigger dots with high contrast color


    def plot_approximation(self, components: List[Callable], parameters: np.ndarray):
        x_range = np.linspace(self.min_x, self.max_x, int(abs(self.min_x - self.max_x) * self.graph_density) + 1)
        y_approx = [self.calculate_function(x, parameters, components) for x in x_range]
        plt.plot(x_range, y_approx, label="Approximated Function", color='red')  # High contrast line color


    @staticmethod
    def calculate_function(x: float, parameters: np.ndarray, components: List[Callable]) -> float:
        return sum(param * component(x) for param, component in zip(parameters, components))

    def load_csv(self, file_path: str) -> List[Point]:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            points = [Point(float(row[0]), float(row[1])) for row in reader if len(row) == 2]

        if points:
            self.min_x, self.max_x = min(p.x for p in points), max(p.x for p in points)
        return points


if __name__ == '__main__':
    approximator=FunctionApproximator()
    
for i, count in enumerate(approximator.G_POINT_COUNTS):
    print(f"Optimal coefficients for function G(x) with {count} points:")
    for j, coeff in enumerate(approximator.optimal_coefficients[i]):
        print(f"  Coefficient {chr(97 + j)}: {coeff}")
    print("\n")