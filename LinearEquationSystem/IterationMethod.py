import numpy as np
from numpy import linalg as la
import random


# checkGershgorinCircles checks whether all the Gershgorin circles
# lies inside the unit circle
def checkGershgorinCircles(n, A):
    for i in range(n):
        r = 0
        for j in range(n):
            r += abs(A[i, j])
        r -= abs(A[i, i])
        if abs(float(A[i, i])) + r > 1:
            return 0
    return 1

# return approximated solution of the linear equation system Ax = b
# if all the eigenvalues are less then 1 by module
# otherwise method possibly doesn't coverge and 0 is returned
def iterationMethod(n, A, b, eps, maxIterations = 1000):
    A = np.eye(n, n) - A
    smallEigenvalues = 1
    if not checkGershgorinCircles(n, A):
        # then there maybe eigenvalue with module greater than 1
        smallEigenvalues = 0
    cnt = 0
    x = np.array([random.random() for i in range(n)]).reshape((n, 1))
    for it in range(maxIterations):
        if la.norm(x - A.dot(x) - b, 2) < eps:
            return x
        xNew = A.dot(x) + b
        if la.norm(xNew) >= la.norm(x) + 1:
            cnt += 1
        else:
            cnt = 0
        if cnt == 20 and smallEigenvalues:
            break
        x = xNew
    return None

def user():
    n = int(input())
    A = np.fromiter(sum([list(map(float, input().split())) for i in range(n)], []), np.float).reshape((n, n))
    b = np.fromiter(list(map(float, input().split())), np.float).reshape((n, 1))
    eps = float(input())
    print(iterationMethod(n, A, b, eps))
