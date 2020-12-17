import numpy as np
from QRAlgorithm import ThreeDiagonalization
from QRAlgorithm import WilkinsonShift
from QRAlgorithm import QRAlgo

EPS = 10**(-5)


def index(n, x, y):
    x %= n
    y %= n
    return x + n * y


def getAlpha(n, A, deg):
    A, _ = ThreeDiagonalization.threediagonalization(n, A)
    A, _ = WilkinsonShift.FastQRAlgo(n, A, EPS)
    d = sorted([A[i, i] for i in range(n)])
    if n == 1:
        return abs(d[0]) / deg
    return max(abs(d[0]), abs(d[n - 2])) / deg


def task1(n):
    M = np.zeros((n * n, n * n))
    for x in range(n):
        for y in range(n):
            M[index(n, x, y), index(n, x + 2 * y, y)] += 1
            M[index(n, x, y), index(n, x - 2 * y, y)] += 1
            M[index(n, x, y), index(n, x + (2 * y + 1), y)] += 1
            M[index(n, x, y), index(n, x - (2 * y + 1), y)] += 1
            M[index(n, x, y), index(n, x, y + 2 * x)] += 1
            M[index(n, x, y), index(n, x, y - 2 * x)] += 1
            M[index(n, x, y), index(n, x, y + (2 * x + 1))] += 1
            M[index(n, x, y), index(n, x, y - (2 * x + 1))] += 1
    return getAlpha(n * n, M, 8)


def task2(p):
    M = np.zeros((p + 1, p + 1))
    for i in range(p):
        M[i, (i + 1) % p] += 1
        M[i, (i - 1) % p] += 1
        if i != 0:
            M[i, pow(i, p - 2, p)] += 1
        else:
            M[i, p] += 1
    M[p, p] += 2
    M[p, 0] += 1
    return getAlpha(p + 1, M, 3)


def user():
    for i in range(1, 10):
        print(task1(i))

print(task2(2))
print(task2(3))
print(task2(5))
print(task2(7))
print(task2(11))
