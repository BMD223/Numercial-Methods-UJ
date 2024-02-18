import matplotlib.pyplot as plt
import numpy as np

def iter_power(mat, tol=1e-6):
    size = len(mat)
    vec = [1.0] * size
    err_list = []
    for _ in range(100):
        temp_vec = vec.copy()
        prod = mat_mult(mat, temp_vec)
        norm_val = sum(val ** 2 for val in prod) ** 0.5
        vec = normalize(prod)
        err = sum(abs(a - b) ** 2 for a, b in zip(temp_vec, vec)) ** 0.5
        err_list.append(err)
        if err < tol:
            break
    return norm_val, vec, err_list

def mat_mult(mat, vec):
    return np.dot(mat, vec)

def normalize(vec):
    norm_val = sum(x ** 2 for x in vec) ** 0.5
    return [x / norm_val for x in vec]

def QR_algo(mat):
    A = np.array(mat)
    diag_list = []
    for _ in range(1, 120 + 1):
        Q, R = np.linalg.qr(A)
        A = np.dot(R, Q)
        eig_vals = np.linalg.eigvals(A)
        err = np.abs(eig_vals - np.diag(A))
        diag_list.append(err)
        if np.all(np.abs(np.diag(A)) < 1e-10):
            break
    return diag_list, eig_vals

def Wilkinson_algo(mat, tol=1e-6):
    size = len(mat)
    vec = [1.0] * size
    eig_vals_list = []

    for _ in range(150):
        temp_vec = vec.copy()
        prod = mat_mult(mat, temp_vec)
        norm_val = sum(val ** 2 for val in prod) ** 0.5
        vec = normalize(prod)
        another_eig_val = sum(vec[i] * sum(mat[i][j] * vec[j] for j in range(size)) for i in range(size)) / sum(
            vec[i] ** 2 for i in range(size))
        eig_vals_list.append(0.5 * (another_eig_val + eigenvalue))
        if norm_val < tol:
            break
    return eig_vals_list

mat_A = [
    [8, 1, 0, 0],
    [1, 7, 2, 0],
    [0, 2, 6, 3],
    [0, 0, 3, 5]
]

eigenvalue, eigenvector, errors = iter_power(mat_A)
print("Przykład a) Dominująca wartość własna:", eigenvalue)
print("Wektor własny:", eigenvector)

errors, eigenvalues = QR_algo(mat_A)
errors = np.array(errors)
colors=['red','grey','green','blue']
for i in range(len(mat_A)):
    plt.plot(range(1, len(errors) + 1), errors[:, i], label=f"Indeks w macierzy {i}:{i}",color=colors[i] )
plt.yscale('log')
plt.xlabel('Iteracje')
plt.ylabel('Różnica')
plt.title('Eigenvalue-przekatna w A')
plt.legend()
plt.show()
print("Przykład b) Wartości własne:", eigenvalues)

eigenvalue, eigenvector, errors = iter_power(mat_A)
plt.plot(range(1, len(errors) + 1), errors, marker='.', label='Metoda Potęgowa')

p_vals = Wilkinson_algo(mat_A)
for i in range(len(mat_A)):
    mat_A[i][i] = abs(mat_A[i][i] - p_vals[i])
eigenvalue_1, eigenvector_1, errors_1 = iter_power(mat_A)
plt.plot(range(1, len(errors_1) + 1), errors_1, marker='.', label='Metoda Wilkinsona')

plt.grid(True)
plt.yscale('log')
plt.xlabel('Iteracje')
plt.ylabel('Błąd zbieżności')
plt.title('Zbieżność')
plt.legend()
plt.show()
