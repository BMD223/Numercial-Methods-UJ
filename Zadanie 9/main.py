import math
import matplotlib.pyplot as plt


def f(x):
    return math.sin(x) - 0.4

def g(x):
    return (math.sin(x) - 0.4) ** 2

def df(x):
    return math.cos(x)

def dg(x):
    sin_x = math.sin(x)
    cos_x = math.cos(x)
    return 2 * (sin_x - 0.4) * cos_x

def u(x):
    denominator = dg(x)
    return g(x) / denominator

def du(x):

    return (5 - 2 * math.sin(x)) / (10 * (math.cos(x) ** 2))


x_star = math.asin(0.4)


x0 = 0
x1 = math.pi / 2
tol = 1e-16
tol2 = 1e-7

def secant_method(f, x0, x1, tol):
    num_steps = 0
    x_values = []
    while True:
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        x_values.append(abs(x2 - x_star))
        if abs(f(x2)) < tol:
            return x2, num_steps, x_values
        x0, x1 = x1, x2
        num_steps += 1


def newton_method(f, df, x0, tol):
    num_steps = 0
    x_values = []
    while True:
        x1 = x0 - f(x0) / df(x0)
        x_values.append(abs(x1 - x_star))
        if abs(f(x1)) < tol or abs(x1 - x0) < tol:
            return x1, num_steps, x_values
        x0 = x1
        num_steps += 1


def bisection_method(f, a, b, tol):
    steps = 0
    x_vals = []
    while True:
        m = (a + b) / 2
        x_vals.append(m)
        steps += 1
        fm = f(m)
        if abs(fm) < tol:
            return m, steps, x_vals
        elif f(a) * fm < 0:
            b = m
        else:
            a = m


def falsi_method(f, a, b, tol):
    steps = 0
    x_vals = []
    while True:
        fa = f(a)
        fb = f(b)
        x = a - fa * (b - a) / (fb - fa)
        x_vals.append(x)
        steps += 1
        fx = f(x)
        if abs(fx) < tol:
            return x, steps, x_vals
        elif fa * fx < 0:
            b = x
        else:
            a = x


r_sec_f, s_sec_f, xvf_sec_f = secant_method(f, x0, x1, tol)
r_newt_f, s_newt_f, xvn_f = newton_method(f, df, x0, tol)
r_bis_f, s_bis_f, xvb_f = bisection_method(f, x0, x1, tol)
r_fal_f, s_fal_f, xvfal_f = falsi_method(f, x0, x1, tol)

xvfal_f = [abs(prev_val - x_star) for prev_val in xvfal_f]
xvb_f = [abs(prev_val - x_star) for prev_val in xvb_f]

plt.figure(figsize=(10, 6))
plt.plot(range(len(xvf_sec_f)), xvf_sec_f, label='Secant (f(x))')
plt.plot(range(len(xvn_f)), xvn_f, label='Newton (f(x))')
plt.plot(range(len(xvb_f)), xvb_f, label='Bisection (f(x))')
plt.plot(range(len(xvfal_f)), xvfal_f, label='Falsi (f(x))')
plt.grid()
plt.yscale('log')
plt.xlabel('Liczba iteracji')
plt.ylabel('Wartość bezw. róznicy |x_i - x*|')
plt.legend()
plt.title('Porównanie dla f(x)')
plt.show()

r_sec_g, s_sec_g, xvg_sec_g = secant_method(g, x0, x1/2, tol)
r_newt_g, s_newt_g, xvn_g = newton_method(g, dg, 0, tol)

plt.figure(figsize=(10, 6))
plt.plot(range(len(xvg_sec_g)), xvg_sec_g, label='Secant (g(x))')
plt.plot(range(len(xvn_g)), xvn_g, label='Newton (g(x))')
plt.yscale('log')
plt.xlabel('Liczba iteracji')
plt.ylabel('Wartość bezw. róznicy |xi - x*|')
plt.legend()
plt.title('Porównanie dla g(x)')
plt.show()



r_sec_u, s_sec_u, xvu_sec_u = secant_method(u, x0, x1, tol2)
r_newt_u, s_newt_u, xvn_u = newton_method(u, du, 0, tol2)
r_bis_u, s_bis_u, xvb_u = bisection_method(u, x0, x1, tol2)

xvb_u = [abs(prev_val - x_star) for prev_val in xvb_u]

def check_finite(values):
    return all(math.isfinite(val) for val in values)

if check_finite(xvu_sec_u) and check_finite(xvn_u) and check_finite(xvb_u):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(xvu_sec_u)), xvu_sec_u, label='Secant (u(x))')
    plt.plot(range(len(xvn_u)), xvn_u, label='Newton (u(x))')
    plt.plot(range(len(xvb_u)), xvb_u, label='Bisection (u(x))')
    plt.yscale('log')
    plt.xlabel('Liczba iteracji')
    plt.ylabel('Wartość bezw. róznicy |xi - x*|')
    plt.legend()
    plt.title('Porównanie dla u(x)')
    plt.show()
else:
    print("ERROR! NIESKONCZONOSC W: x_values.")