import numpy as np

def GaussSeidel(A,B,n):
    #A: la matrice da risolvere
    #B: il termine noto
    #n: il numero di iterazioni
    x = np.zeros(np.shape(B))
    L = np.tril(A)
    U = A-L
    for i in range(n):
        x = np.dot(np.linalg.inv(L),B-np.dot(U,x))
    return x

