import numpy as np
import random
from QRAlgorithm import IterationMethod
from QRAlgorithm import ThreeDiagonalization
from QRAlgorithm import WilkinsonShift

EPS = 10 ** (-4)


def checkMaxEigenvalue(n, A, B):
    for i in range(20):
        x0 = np.array([random.random() for i in range(n)]).reshape((n, 1))
        l1, v1 = IterationMethod.iterationMethod(A, x0, EPS)
        l2, v2 = IterationMethod.iterationMethod(B, x0, EPS)
        if (l1 is not None) and (l2 is not None):
            if abs(l1 - l2) > EPS:
                return 0
            else:
                break
    return 1


def checkDegrees(n, A, B):
    a = sorted([A[i, ...].sum() for i in range(n)])
    b = sorted([B[i, ...].sum() for i in range(n)])
    return a == b


def dfs(v, M, n, used):
    cnt = 0
    used[v] = 1
    for i in range(n):
        if M[v, i] == 1 and not used[i]:
           cnt += dfs(i, M, n, used)
    return cnt + 1


def checkComponents(n, A, B):
    usedA = [0 for i in range(n)]
    usedB = [0 for i in range(n)]
    a = []
    b = []
    for i in range(n):
        if not usedA[i]:
            a.append(dfs(i, A, n, usedA))
        if not usedB[i]:
            b.append(dfs(i, B, n, usedB))
    a = sorted(a)
    b = sorted(b)
    return a == b


def isomorphic(n, A, m, B):
    if n != m:
        return 0
    if A.sum() != B.sum():  # if numbers of edges are different
        return 0
    if not checkDegrees(n, A, B):  # if sets of degrees of vertexes are different
        return 0
    if not checkComponents(n, A, B):  # if sets of numbers of vertexes in connectivity components are different
        return 0
    if checkMaxEigenvalue(n, A, B) == 0:
        return 0
    A, _ = ThreeDiagonalization.threediagonalization(n, A)
    B, _ = ThreeDiagonalization.threediagonalization(n, B)
    A, _ = WilkinsonShift.FastQRAlgo(n, A, EPS)
    B, _ = WilkinsonShift.FastQRAlgo(n, B, EPS)
    if (A is not None) and (B is not None):
        a = sorted([A[i, i] for i in range(n)])
        b = sorted([B[i, i] for i in range(n)])
        for i in range(n):
            if abs(a[i] - b[i]) > EPS:
                return 0
    return 1


def user():
    n = int(input())
    A = np.fromiter(sum([list(map(float, input().split())) for i in range(n)], []), np.float).reshape((n, n))
    m = int(input())
    B = np.fromiter(sum([list(map(float, input().split())) for i in range(m)], []), np.float).reshape((m, m))
    print(isomorphic(n, A, m, B))

