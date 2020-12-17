import numpy as np


EPS = 10**(-5)


def GivensRotation(A, i, j, c, s):
    u = c * A[i, ...] + s * A[j, ...]
    v = -s * A[i, ...] + c * A[j, ...]
    A[i, ...] = u
    A[j, ...] = v
    return 0

# for threediagonal matrices this algorithm time complexity will be O(n^2)
# as there're not more than 3 non-zero elements in a column
# that's why GivensRotation function will be called O(n) times
def QRDecomposition(n, A):
    Q = np.eye(n)
    R = np.copy(A)
    decomposition = [] # Q is the production of decomposition elements in reversed order
                       # decompostition elements are Givens matrixes
    for i in range(n):
        for j in range(i + 1, n):
            if abs(R[j, i]) < EPS:
                continue
            c = R[i, i] / ((R[i, i] ** 2 + R[j, i] ** 2) ** 0.5)
            s = R[j, i] / ((R[i, i] ** 2 + R[j, i] ** 2) ** 0.5)
            GivensRotation(Q, i, j, c, s)
            GivensRotation(R, i, j, c, s)
            decomposition.append((i, j, c, s))
    return Q.T, R, decomposition


# fastMultiplication calculates product A * Q where Q = (Gk * ... * G0)^T
# Gi - Givens' matrices from decomposition list
# idea: B = A * Q = ((A * Q)^T)^T = (Gk * ... * G0 * A^T)^T
# fastMultiplication has O(n^2 + n * k) time complexity where k = decomposition.size()
# for threediagonal matrices decomposition size is O(n) that's why time complexity will be O(n^2)
def fastMultiplication(A, decomposition):
    B = np.copy(A).T
    for index in range(len(decomposition)):
        i, j, c, s = decomposition[index]
        GivensRotation(B, i, j, c, s)
    return B.T


def user():
    n = int(input())
    A = np.fromiter(sum([list(map(float, input().split())) for i in range(n)], []), np.float).reshape((n, n))
    Q, R, decomposition = QRDecomposition(n, A)
    print(Q.round(5))
    print(R.round(5))