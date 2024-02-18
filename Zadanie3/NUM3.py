import numpy as np
import matplotlib.pyplot as plt
import time

def checkNumpy(matrix_A,matrix_x):
    
    start = time.time()

    np.linalg.solve(matrix_A, matrix_x)

    print(f"Czas numpy to: {time.time()-start}")

def determinant(matrix,n):
    determinant = 1
    for i in range(n):
        determinant *= matrix[i][1]
    return determinant

def program(size):
    
    matrix_x=[i for i in range (1,size+1)]

    matrix_A=[[0,1.2, 0.1, 0.15]]
    for i in range (1,size-2):
        tempList=[0.2,1.2,0.1/(i+1),0.15/((i+1)**2)]
        matrix_A.append(tempList)
    matrix_A.append([0.2,1.2,0.1/size-1,0])
    matrix_A.append([0.2,1.2,0,0])
    
    start = time.time()
    
    for i in range(1, size-2):
        matrix_A[i][0] = matrix_A[i][0] / matrix_A[i-1][1]
        matrix_A[i][1] = matrix_A[i][1] - matrix_A[i][0] * matrix_A[i - 1][2]
        matrix_A[i][2] = matrix_A[i][2] - matrix_A[i][0] * matrix_A[i - 1][3]

    matrix_A[size-2][0] = matrix_A[size-2][0] / matrix_A[size-3][1]
    matrix_A[size-2][1] = matrix_A[size-2][1] - matrix_A[size-2][0] * matrix_A[size-3][2]
    matrix_A[size-2][2] = matrix_A[size-2][2] - matrix_A[size-2][0] * matrix_A[size-3][3]

    matrix_A[size-1][0] = matrix_A[size-1][0] / matrix_A[size-2][1]
    matrix_A[size-1][1] = matrix_A[size-1][1] - matrix_A[size-1][0] * matrix_A[size-2][2]

    for i in range(1, size):
        matrix_x[i] = matrix_x[i] - matrix_A[i][0] * matrix_x[i - 1]


    matrix_x[size-1] = matrix_x[size-1] / matrix_A[size-1][1]
    matrix_x[size-2] = (matrix_x[size-2] - matrix_A[size-2][2] * matrix_x[size-1]) / matrix_A[size-2][1]

    for i in range(size - 3, -1, -1):
        matrix_x[i] = (matrix_x[i] - matrix_A[i][3] * matrix_x[i + 2] - matrix_A[i][2] * matrix_x[i + 1]) / matrix_A[i][1]
    
    return time.time()-start

num_runs = 150
runtimes = []
program(124)

for i in range(124,1000):
    runtime = program(i)
    runtimes.append(runtime)

average_runtime = np.mean(runtimes)
x_values = list(range(50, 50 + len(runtimes)))

min_runtime_idx = np.argmin(runtimes)
max_runtime_idx = np.argmax(runtimes)

plt.plot(x_values,runtimes, label="Czas(sekundy)")
plt.axhline(average_runtime, color='r', linestyle='--', label=f'Średni czas trwania: {average_runtime:.2f}')
plt.plot([min_runtime_idx+50, max_runtime_idx+50], [runtimes[0], runtimes[-1]], 'g--', label="Zalenośc liniowa")
plt.xlabel("Ilość wprowadzonych danych")
plt.ylabel("Czas działania programu")
plt.legend()
plt.title("Wykres czasu do danych")
plt.show()

