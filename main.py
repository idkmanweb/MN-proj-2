import matplotlib.pyplot as mpl
import time
import copy
import math

# index: 193382
n = 982
time_check = False


def dot_product(a, b):
    result = []

    for i in range(len(b)):
        current = 0

        if isinstance(a[i], list):
            for j in range(len(b)):
                current += a[i][j] * b[j]
        else:
            current = a[i] * b[i]

        result.append(current)

    return result


def subtract(a, b):
    result = []

    for i in range(len(b)):
        if isinstance(a[i], list):
            current = []

            for j in range(len(b)):
                current.append(a[i][j] - b[j])

            result += [current]
        else:
            result.append(a[i] - b[i])

    return result


def norm(res):
    result = 0

    for i in range(len(res)):
        result += res[i] ** 2

    return math.sqrt(result)


def create_equations(n):
    a1 = 8
    a2 = -1
    a3 = -1
    f = 3
    b = []

    for i in range(n):
        b.append(math.sin((i + 1) * (f + 1)))

    a = [[0. for _ in range(n)] for _ in range(n)]

    for i in range(n):
        a[i][i] = a1
        if i > 0:
            a[i][i - 1] = a2
        if i > 1:
            a[i][i - 2] = a3
        if i < n - 1:
            a[i][i + 1] = a2
        if i < n - 2:
            a[i][i + 2] = a3

    return a, b


def jacobi_solve(A, b):
    iteration = 0
    calculating = True
    x = [1] * len(b)
    new_x = [1] * len(b)
    ress = []

    while calculating:
        iteration += 1

        for i in range(len(b)):
            sums = 0

            for j in range(0, len(b)):
                if i != j:
                    sums += A[i][j] * x[j]
                if sums != 0 and A[i][j] == 0:
                    break

            new_x[i] = (b[i] - sums) / A[i][i]

        x = copy.copy(new_x)
        dp = dot_product(A, x)
        sub = subtract(dp, b)
        res = norm(sub)
        ress.append(res)

        if res < 10 ** (-9):
            calculating = False

        if res > ress[len(ress) - 2]:
            calculating = False
            print("Metoda jest rozbieżna")

    return x, iteration, ress


def gauss_solve(A, b):
    iteration = 0
    calculating = True
    x = [1] * len(b)
    ress = []

    while calculating:
        iteration += 1

        for i in range(len(b)):
            sums = 0

            for j in range(0, len(b)):
                if i != j:
                    sums += A[i][j] * x[j]
                if sums != 0 and A[i][j] == 0:
                    break

            x[i] = (b[i] - sums) / A[i][i]

        res = norm(subtract(dot_product(A, x), b))
        ress.append(res)

        if res < 10 ** (-9):
            calculating = False

        if res > ress[len(ress) - 2]:
            calculating = False
            print("Metoda jest rozbieżna")

    return x, iteration, ress


def lu_factorization(A, b):
    L = [[0.0] * len(b) for _ in range(len(b))]
    U = [[0.0] * len(b) for _ in range(len(b))]

    for i in range(len(b)):
        for j in range(i, len(b)):
            sum = 0.0
            for k in range(i):
                sum += (L[i][k] * U[k][j])
            U[i][j] = A[i][j] - sum

        for j in range(i, len(b)):
            if i == j:
                L[i][i] = 1.0
            else:
                sum = 0.0
                for k in range(i):
                    sum += (L[j][k] * U[k][i])
                L[j][i] = (A[j][i] - sum) / U[i][i]

    y = [0.] * len(b)

    for i in range(len(b)):
        sum = 0.
        for j in range(i):
            sum += L[i][j] * y[j]
        y[i] = (b[i] - sum)

    x = [0.] * len(b)

    for i in reversed(range(len(b))):
        sum = 0.
        for j in range(i + 1, len(b)):
            sum += U[i][j] * x[j]
        x[i] = (y[i] - sum) / U[i][i]

    return x


if time_check == False:
    A, b = create_equations(n)

    start = time.time()
    x, iterations_jacobi, res_jacobi = jacobi_solve(A, b)
    end = time.time()
    print("Liczba iteracji w metodzie Jacobiego: " + str(iterations_jacobi))
    print("Czas metody Jacobiego: " + str(end - start))

    print()

    start = time.time()
    x, iterations_gauss, res_gauss = gauss_solve(A, b)
    end = time.time()
    print("Liczba iteracji w metodzie Gaussa-Seidla: " + str(iterations_gauss))
    print("Czas metody Gaussa-Seidla: " + str(end - start))

    mpl.yscale("log")
    mpl.plot(range(iterations_jacobi), res_jacobi, color='r')
    mpl.plot(range(iterations_gauss), res_gauss, color='g')
    mpl.legend(["Metoda Jacobiego", "Metoda Gaussa-Seidla"])
    mpl.title("Norma residuum w każdej iteracji")
    mpl.ylabel("Norma residuum")
    mpl.xlabel("Iteracja")
    mpl.show()

    print()

    start = time.time()
    x = lu_factorization(A, b)
    end = time.time()
    print("Czas faktoryzacji LU: " + str(end - start))

    res = norm(subtract(dot_product(A, x), b))
    print("Norma residuum faktoryzacji LU: " + str(res))

else:
    N = [100, 500, 1000, 2000, 3000]
    jacobi_times = []
    gauss_times = []
    lu_times = []

    for n in N:
        A, b = create_equations(n)

        start = time.time()
        x, iterations_jacobi, res_jacobi = jacobi_solve(A, b)
        end = time.time()
        time_jacobi = end - start

        start = time.time()
        x, iterations_gauss, res_gauss = gauss_solve(A, b)
        end = time.time()
        time_gauss = end - start

        start = time.time()
        x = lu_factorization(A, b)
        end = time.time()
        time_lu = end - start

        jacobi_times.append(time_jacobi)
        gauss_times.append(time_gauss)
        lu_times.append(time_lu)

    mpl.plot(N, jacobi_times, color='r')
    mpl.plot(N, gauss_times, color='g')
    mpl.plot(N, lu_times, color='b')
    mpl.legend(["Metoda Jacobiego", "Metoda Gaussa-Seidla", "Faktoryzacja LU"])
    mpl.title("Czas rozwiązania zależnie od ilości niewiadomych")
    mpl.ylabel("Czas wykonania")
    mpl.xlabel("Ilość niewiadomych")
    mpl.show()
