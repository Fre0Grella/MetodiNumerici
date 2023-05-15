import numpy as np
from scipy.linalg import lu

def gauss_seidel(A,b,n,tollerance,x=None):
    """Solves the equation Ax=b via the Gauss-Seidel iterative method."""
    # Create an initial guess if needed
    if x is None:
        x = np.zeros(np.shape(b))
    
    L = np.tril(A)
    U = A-L
    errore = 1000
    i = 0
    while i <= n and errore >= tollerance:
        x0=x.copy()
        x = np.dot(np.linalg.inv(L),b-np.dot(U,x))
        errore=np.linalg.norm(x-x0)/np.linalg.norm(x)
        i=i+1
    return x



def jacobi(A,b,n,tollerance,x=None):
    """Solves the equation Ax=b via the Jacobi iterative method."""
    # Create an initial guess if needed                                                                                                                                                            
    if x is None:
        x = np.zeros(np.shape(b))
                                                                                                                                                                   
    D = np.diag(A)
    R = A - np.diagflat(D)
                                                                                                                                                                         
    errore = 1000
    i = 0
    while i <= n and errore >= tollerance:
        x0=x.copy()
        x = (b - np.dot(R,x)) / D
        errore=np.linalg.norm(x-x0)/np.linalg.norm(x)
        i=i+1
    return x



def sor_solver(A, b, w,n, tollerance, x=None):
    """Solves the equation Ax=b via Succesive Over Relaxation Method."""

    """ w: relaxation factor (typically 0 < w < 2)
        tollerance: the amount of error accepted
    """
    # Create an initial guess if needed                                                                                                                                                            
    if x is None:
        x = np.zeros(np.shape(b))

    residual = np.linalg.norm(np.matmul(A, x) - b)
    i=0
    while residual >= tollerance and i <= n:
        for i in range(A.shape[0]):
            sigma = 0
            for j in range(A.shape[1]):
                if j != i:
                    sigma += A[i][j] * x[j]
            x[i] = (1 - w) * x[i] + (w / A[i][i]) * (b[i] - sigma)
        residual = np.linalg.norm(np.matmul(A, x) - b)
        #print('Residual: {0:10.6g}'.format(residual))
        i=i+1
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


def Lsolve(L,b):
    """  
    Risoluzione con procedura forward di Lx=b con L triangolare inferiore  
     Input: L matrice triangolare inferiore
            b termine noto
    Output: x: soluzione del sistema lineare
            flag=  0, se sono soddisfatti i test di applicabilità
                   1, se non sono soddisfatti
    """
#test dimensione
    m,n=L.shape
    flag=0
    if n != m:
        print('errore: matrice non quadrata')
        flag=1
        x=[]
        return x, flag
    
     # Test singolarita'
    if np.all(np.diag(L)) != True:
         print('el. diag. nullo - matrice triangolare inferiore')
         x=[]
         flag=1
         return x, flag
    # Preallocazione vettore soluzione
    x=np.zeros((n,1))
    
    for i in range(n):
         s=np.dot(L[i,:i],x[:i]) #scalare=vettore riga * vettore colonna
         x[i]=(b[i]-s)/L[i,i]
      
     
    return x,flag

def Usolve(U,b):
    
    """
    Risoluzione con procedura backward di Rx=b con R triangolare superiore  
     Input: U matrice triangolare superiore
            b termine noto
    Output: x: soluzione del sistema lineare
            flag=  0, se sono soddisfatti i test di applicabilità
                   1, se non sono soddisfatti
    
    """ 
#test dimensione
    m,n=U.shape
    flag=0
    if n != m:
        print('errore: matrice non quadrata')
        flag=1
        x=[]
        return x, flag
    
     # Test singolarita'
    if np.all(np.diag(U)) != True:
         print('el. diag. nullo - matrice triangolare superiore')
         x=[]
         flag=1
         return x, flag
    # Preallocazione vettore soluzione
    x=np.zeros((n,1))
    
    for i in range(n-1,-1,-1):
         s=np.dot(U[i,i+1:n],x[i+1:n]) #scalare=vettore riga * vettore colonna
         x[i]=(b[i]-s)/U[i,i]
      
     
    return x,flag

def LUsolve(P,A,L,U,b):
    pb=np.dot(P,b)
    y,flag=Lsolve(L,pb)
    if flag == 0:
         x,flag=Usolve(U,y)
    else:
        return [],flag

    return x,flag


def Risoluzione_QR(A,b):
    '''
    Non è una funzione da usare serve solo come reminder
    '''
    Q,R=np.linalg.qr(A)
    y=np.dot(Q.T,b)
    xqr,flag=Usolve(R,y)
    return xqr, flag
    


def Risoluzione_LU(A,b):
    '''
    Non è una funzione da usare serve solo come reminder
    '''
    PT, L, U = lu(A)
    P=PT.T.copy()
    #Le permutazioni di righe fatte sulla matrice vengono effettuate anche sul termine noto
    x,flag=LUsolve(P,A,L,U,b)
    return x, flag
