import numpy as np
from scipy.linalg import lu

def gauss_seidel(A,b,n,x=None):
    """Solves the equation Ax=b via the Gauss-Seidel iterative method."""
    # Create an initial guess if needed
    if x is None:
        x = np.zeros(np.shape(b))
    
    L = np.tril(A)
    U = A-L
    for i in range(n):
        x = np.dot(np.linalg.inv(L),b-np.dot(U,x))
    return x



def jacobi(A,b,n=50,x=None):
    """Solves the equation Ax=b via the Jacobi iterative method."""
    # Create an initial guess if needed                                                                                                                                                            
    if x is None:
        x = np.zeros(np.shape(b))
                                                                                                                                                                   
    D = np.diag(A)
    R = A - np.diagflat(D)
                                                                                                                                                                         
    for i in range(n):
        x = (b - np.dot(R,x)) / D
    return x



def sor_solver(A, b, w, tollerance, x=None):
    """Solves the equation Ax=b via Succesive Over Relaxation Method."""

    """ w: relaxation factor (typically 0 < w < 2)
        tollerance: the amount of error accepted
    """
    # Create an initial guess if needed                                                                                                                                                            
    if x is None:
        x = np.zeros(np.shape(b))

    residual = np.linalg.norm(np.matmul(A, x) - b)
    while residual > tollerance:
        for i in range(A.shape[0]):
            sigma = 0
            for j in range(A.shape[1]):
                if j != i:
                    sigma += A[i][j] * x[j]
            x[i] = (1 - w) * x[i] + (w / A[i][i]) * (b[i] - sigma)
        residual = np.linalg.norm(np.matmul(A, x) - b)
        #print('Residual: {0:10.6g}'.format(residual))
    return x

def least_squares(A, b):
    """Solves the equation Ax=b via Least Squares Method."""

    n = A.shape[1]
    r = np.linalg.matrix_rank(A)
    U, sigma, VT = np.linalg.svd(A, full_matrices=True)
    D_plus = np.diag(np.hstack([1/sigma[:r], np.zeros(n-r)]))
    V = VT.T
    A_plus = V.dot(D_plus).dot(U.T)
    w = A_plus.dot(b)
    error = np.linalg.norm(A.dot(w) - b, ord=2) ** 2
    return w,error

def LU(A):
    PT,L,U=lu(A)  #Restituisce in output la trasposta della matrice di Permutazione
    P=PT.T.copy()   #P è la matrice di permutazione
    print("A=",A)
    print("L=",L)
    print("U=",U)
    print("P=",P)
    #LU è la fattorizzazione di P*A (terorema 2)
    A2=P@A # equivale al prodotto matrice x matrice np.dot(P,A)
    A2Fatt=L@U # equivale a np.dot(L,U)
    print("Matrice P*A \n", A2)
    print("Matrice ottenuta moltipicando Le ed U \n",A2Fatt)