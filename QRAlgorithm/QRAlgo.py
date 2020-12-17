import numpy as np
from QRDecomposition import GivensRotation
from QRDecomposition import HouseholderTransformation

def isDiag(n, A, eps):
    for i in range(n):
        r = 0
        for j in range(n):
            r += abs(A[i, j])
        r -= abs(A[i, i])
        if r > eps:
            return 0
    return 1


# QRAlgo step takes O(n^3) time for arbitrary matrices and O(n^2) for threediagonal
def QRAlgo(n, A, eps, maxIterations=10000):
    A_0 = np.copy(A)
    Qk = np.eye(n)
    for i in range(maxIterations):
        if isDiag(n, A, eps):
            return A, Qk
        Q, R, decomposition = GivensRotation.QRDecomposition(n, A) # here you can use Householder decomposition
                                                                   # to calculate QR decomposition
        Qk = GivensRotation.fastMultiplication(Qk, decomposition)
        A = GivensRotation.fastMultiplication(R, decomposition)
    return A, Qk


def user():
    n = int(input())
    A = np.fromiter(sum([list(map(float, input().split())) for i in range(n)], []), np.float).reshape((n, n))
    eps = float(input())
    D, Q = QRAlgo(n, A, eps)
    print(D.round(4))
    print(Q.round(5))

#user()