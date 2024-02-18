import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix

N = 124
value = 1000.0
A = np.diag([3] * N) + np.diag([1] * (N - 1), k=1) + np.diag([1] * (N - 1), k=-1) + \
    np.diag([0.15] * (N - 2), k=2) + np.diag([0.15] * (N - 2), k=-2)
b = np.array([2] + [3] * (N - 2) + [N])
x0 = np.full(N, value)

def jacobi(A, b, x0=None, epsilon=1e-10, max_iterations=1000):
    D = np.diag(np.diag(A))
    R = A - D
    invd = np.linalg.inv(D)
    xj = x0 if x0 is not None else np.zeros_like(b)
    errors_j = []

    for _ in range(max_iterations):
        x_new = invd @ (b - R @ xj)
        errors_j.append(np.linalg.norm(x_new - xj))
        if np.linalg.norm(x_new - xj) < epsilon:
            break
        xj = x_new

    return xj, errors_j

def gauss_seidel(A, b, x0=None, epsilon=1e-10, max_iterations=1000):
    xgs = x0 if x0 is not None else np.zeros_like(b)
    errors_gs = []

    for _ in range(max_iterations):
        x_prev = np.copy(xgs)
        for i in range(len(A)):
            sum1 = np.dot(A[i, :i], xgs[:i])
            sum2 = np.dot(A[i, i + 1:], xgs[i + 1:])
            xgs[i] = (b[i] - sum1 - sum2) / A[i, i]
        errors_gs.append(np.linalg.norm(xgs - x_prev))
        if np.linalg.norm(xgs - x_prev) < epsilon:
            break
        x_prev = np.copy(xgs)

    return xgs, errors_gs

# Dokladne rozwiazanie
A_csc = csc_matrix(A)
x_exact = spsolve(A_csc, b)

xj, errors_j = jacobi(A, b, x0)
xgs, errors_gs = gauss_seidel(A, b, x0)

print("Sprawdzone rozwiązanie:", x_exact[:5])
print("Rozwiązanie Jacobiego:", xj[:5])
print("Rozwiązanie Gaussa-Seidela:", xgs[:5])

# Graf
plt.figure(figsize=(12, 6))
plt.semilogy(errors_j, label='Metoda Jacobiego', c="red")
plt.semilogy(errors_gs, label='Metoda Gaussa-Seidela', c="green")
plt.xlabel('Iteracje')
plt.ylabel('Błąd (logarytmiczna skala)')
plt.title('Zbieżność metod')
plt.legend()
plt.grid(True)
plt.show()
