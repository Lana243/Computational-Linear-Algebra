import numpy as np
from numpy import linalg as la
import random


# return approximated solution of the linear equation system Ax = b
# if numbers on diagonal are not zeros and method converges
# otherwise 0 is returned
def GaussZeidelMethod(n, A, b, eps, maxIterations = 1000):
    L = np.array([[A[i][j] if i >= j else 0 for j in range(n)] for i in range(n)]).reshape((n, n))
    U = A - L
    for i in range(n):
        if (A[i, i] == 0):
            return None
    cnt = 0
    x = np.array([random.random() for i in range(n)]).reshape((n, 1))
    for it in range(maxIterations):
        if la.norm(A.dot(x) - b) < eps:
            return x
        c = -U.dot(x) + b
        xNew = np.zeros((n, 1))
        for i in range(n):
            s = c[i] - L[i, ...].dot(xNew)
            xNew[i, 0] = s / A[i, i]
        if la.norm(xNew) >= la.norm(x) + 1:
            cnt += 1
        else:
            cnt = 0
        if cnt == 20:
            break
        x = xNew
    return None


def user():
    n = int(input())
    A = np.fromiter(sum([list(map(float, input().split())) for i in range(n)], []), np.float).reshape((n, n))
    b = np.fromiter(list(map(float, input().split())), np.float).reshape((n, 1))
    eps = float(input())
    print(GaussZeidelMethod(n, A, b, eps))
