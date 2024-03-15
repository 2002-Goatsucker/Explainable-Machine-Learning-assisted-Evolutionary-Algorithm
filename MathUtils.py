from dependency import *
def LatinHypercube(left, right, sample_num, dim):
    return left + (right-left) * lhs(dim, sample_num)

def calculateAll(X, func):
    y = []
    for x in X:
        y.append(func(np.array([x], dtype=float)))
    return np.array(y)