import numpy as np
from QRDecomposition import HouseholderTransformation
from numpy import linalg as la


def threediagonalization(n, A, eps=10**(-5)):
    Q = np.eye(n)
    newA = A
    for i in range(n - 1):
        v = newA[(i + 1):, i]
        if la.norm(v) < eps:
            continue
        u = v / la.norm(v)
        u[0] -= 1
        if la.norm(u) < eps:
            continue
        u = u / la.norm(u)
        a = [0 if j <= i else u[j - i - 1] for j in range(n)]
        v = np.array(list(a))
        Q = HouseholderTransformation.HouseholderTransformationRight(n, Q, v)
        newA = HouseholderTransformation.HouseholderTransformationLeft(n, newA, v)
        newA = HouseholderTransformation.HouseholderTransformationRight(n, newA, v)
    return newA, Q


def user():
    n = int(input())
    A = np.fromiter(sum([list(map(float, input().split())) for i in range(n)], []), np.float).reshape((n, n))
    newA, Q = threediagonalization(n, A)
    print(newA.round(5))
    print(Q.round(5))

