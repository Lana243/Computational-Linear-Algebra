import numpy as np
from numpy import linalg as la

EPS = 10**(-9)


def iterationMethod(A, x0, eps, maxIterations=1000):
    if la.norm(x0) < EPS:
        return None, None
    x = x0 / la.norm(x0)
    for i in range(maxIterations):
        v = A.dot(x)
        l = (x.T).dot(v)
        if la.norm(v - l * x) < eps:
            return l, x
        x = A.dot(x)
        if la.norm(x) < EPS:
            return None, None
        x = x / la.norm(x)
    return None, None


def user():
    n = int(input())
    A = np.fromiter(sum([list(map(float, input().split())) for i in range(n)], []), np.float).reshape((n, n))
    x0 = np.fromiter(list(map(float, input().split())), np.float).reshape((n, 1))
    eps = float(input())
    l, v = iterationMethod(A, x0, eps)
    print(l)
    print(v)

user()