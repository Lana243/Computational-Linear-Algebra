import numpy as np
from QRAlgorithm import QRAlgo
from QRDecomposition import GivensRotation
from QRDecomposition import HouseholderTransformation


def GershgorinCheck(n, A, i, eps):
    row = 0
    column = 0
    for j in range(n):
        row += abs(A[i, j])
        column += abs(A[j, i])
    row -= abs(A[i, i])
    column -= abs(A[i, i])
    return row < eps and column < eps


def FastQRAlgo(n, A, eps, maxIterations=1000):
    Qk = np.eye(n)
    E = np.eye(n)
    for i in range(n - 1, 0, -1):
        for j in range(maxIterations):
            if GershgorinCheck(n, A, i, eps):
                break
            a = A[i - 1, i - 1]
            b = A[i - 1, i]
            c = A[i, i]
            e0 = 0.5 * (a + c + ((a - c)**2 + 4 * b**2)**(0.5))
            e1 = 0.5 * (a + c - ((a - c)**2 + 4 * b**2)**(0.5))
            s = e0
            if abs(e1 - A[i, i]) < abs(e0 - A[i, i]):
                s = e1
            Q, R, _ = GivensRotation.QRDecomposition(n, A - s * E)
            Q, R, decomposition = GivensRotation.QRDecomposition(n, A - s * E)
            A = GivensRotation.fastMultiplication(R, decomposition) + s * E
            Qk = GivensRotation.fastMultiplication(Qk, decomposition)
    return A, Qk


def user():
    n = int(input())
    A = np.fromiter(sum([list(map(float, input().split())) for i in range(n)], []), np.float).reshape((n, n))
    eps = float(input())
    A, Qk = FastQRAlgo(n, A, eps)
    np.set_printoptions(precision=3)
    print(A)
    print(Qk)
