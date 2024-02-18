import time
import numpy as np           
import matplotlib.pyplot as plt

size = 80
matrix_b = [5]*size

def checknumpy():
    matrix_A = np.ones((size, size))
    matrix_A += np.diag([11] * size)
    matrix_A += np.diag([7] * (size - 1), 1)
    start = time.time()
    np.linalg.solve(matrix_A, matrix_b)
    return time.time()-start

def solution():
    matrix_A=[]
    matrix_A.append([11] * size)
    matrix_A.append([7] * (size - 1) + [0])
    
    start = time.time()

    z = [0]*size
    x = [0]*size
    z[size-1] = matrix_b[size-1] / matrix_A[0][size-1]
    x[size-1] = 1 / matrix_A[0][size-1]
    
    for i in range(size - 2, -1, -1):
        z[i] = (matrix_b[size-2] - matrix_A[1][i] * z[i+1]) / matrix_A[0][i]
        x[i] = (1 - matrix_A[1][i] * x[i+1]) / matrix_A[0][i]
    
    delta = sum(z)/(1+sum(x))
    result=[]
    for i in range(len(z)):
        result.append(z[i]-x[i]*delta)
    #print(result)
    return time.time()-start
solution()
checknumpy()

N = []
numpy_time = []
real = []
for i in range(80, 3000,20):
    size = i
    N.append(size)
    matrix_b = [5] * size
    numpy_time.append(checknumpy() * 1000)
    real.append(solution() * 1000)
plt.yscale('log')
plt.title('Zaleznosc ilosci elementow do czasu pracy')
plt.xlabel('Ilosc elementow')
plt.ylabel('czas w ms')
plt.legend(['Czas Numpy', 'Czas pracy algorytmu'])
plt.plot(N, numpy_time, c='blue',label="Czas dla numpy")
plt.plot(N, real, c='red',label="Czas dla Shermana")
plt.legend()
plt.show()