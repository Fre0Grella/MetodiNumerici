import numpy as np

def gauss_seidel(A,b,n=50,x=None):
    """Solves the equation Ax=b via the Gauss-Seidel iterative method."""
    # Create an initial guess if needed
    if x is None:
        x = np.zeros(len(A[0]))
    
    L = np.tril(A)
    U = A-L

    for i in range(n):
        x = np.dot(np.linalg.inv(L),b-np.dot(U,x))
    return x

def jacobi(A,b,n=50,x=None):
    """Solves the equation Ax=b via the Jacobi iterative method."""
    # Create an initial guess if needed                                                                                                                                                            
    if x is None:
        x = np.zeros(len(A[0]))
                                                                                                                                                                   
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
        x = np.zeros(len(A[0]))

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