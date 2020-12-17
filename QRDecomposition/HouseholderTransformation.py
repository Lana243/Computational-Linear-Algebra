import numpy as np
from numpy import linalg as la


def HouseholderTransformationLeft(n, A, v):
    v = v.reshape((n, 1))
    B = A - (2 * v).dot(v.T.dot(A))
    return B


def HouseholderTransformationRight(n, A, v):
    v.reshape((n, 1))
    B = A - (2 * (A.dot(v).reshape((n, 1))).dot(v.reshape(1, n)))
    return B


def QRDecomposition(n, A, eps=10**(-5)):
    Q = np.eye(n)
    R = A
    for i in range(n):
        v = R[i:, i]
        if la.norm(v) < eps:
            continue
        u = v / la.norm(v)
        u[0] -= 1
        if la.norm(u) < eps:
            continue
        u = u / la.norm(u)
        a = [0 if j < i else u[j - i] for j in range(n)]
        v = np.array(list(a))
        Q = HouseholderTransformationLeft(n, Q, v)
        R = HouseholderTransformationLeft(n, R, v)
    return Q.T, R


def user():
    n = int(input())
    A = np.fromiter(sum([list(map(float, input().split())) for i in range(n)], []), np.float).reshape((n, n))
    Q, R = QRDecomposition(n, A)
    print(Q.round(5))
    print(R.round(5))
