import numpy as np
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import time

#GLOBAL VARIABLES:
eps = 1e-5
iteration = 1000

#FUNCTIONS
def generate_corrected_sparse_tridiagonal_matrix(n, diagonal_value=5, off_diagonal_value=1):
    """
    Generates a sparse tridiagonal matrix, ensuring no overlaps.

    Args:
        n: Dimension of the system (size of the matrix A).
        diagonal_value: Value for the diagonal elements.
        off_diagonal_value: Value for the off-diagonal elements.

    Returns:
        A: Sparse coefficient matrix (scipy.sparse.csr_matrix).
        b: Right-hand side vector (numpy array).
    """
    x0 = np.zeros(n)
    # Main diagonal
    main_diag = np.full(n, diagonal_value+off_diagonal_value)

    # Construct sparse matrix
    # Construction of data of non-zero scalars
    data = np.zeros(3*n-2)
    data = np.concatenate(([5,1], (n-2)*[1 , 5 , 1], [1, 5]))
    #Construction of rows and columns vector used to build the good-looking matrix(tri diagonal)
    rows = [0,0]
    cols = np.arange(2)
    for i in range(1,n-1):
        rows = np.concatenate((rows, [i,i,i]))
        cols = np.concatenate((cols,np.arange(i-1,i+2)))
    cols = np.concatenate((cols, [n-2, n-1]))
    rows = np.concatenate((rows, [n-1 ,n-1 ]))
   #Creation of the sparse matrix with the scipy built-in function
    As = csr_matrix((data, (rows, cols)), shape=(n,n))

    # Construct dense matrix for reference
    A_dense = As.toarray()
    #creation of an arbitrary b vector as the right-side of the equation
    b = np.zeros_like(x0)
    return As, b

def generate_sparse_tridiagonal_matrix(n):
    """
    Generates a sparse tridiagonal matrix with the specific values.

    Args:
        n: Dimension of the system (size of the matrix A).

    Returns:
        A: Sparse coefficient matrix (scipy.sparse.csr_matrix).
        b: Right-hand side vector (numpy array).
    """
    ### TODO: Fill your code here.
    A = csr_matrix((n, n))

    # Construct dense matrix for reference
    A_dense = np.zeros((n, n))
    
    # Right-hand side vector
    b = np.zeros(n)

    return A,  A_dense, b



def jacobi_sparse_with_error(A, b, x0, x_exact, tol=eps, max_iter=iteration):
    """
    Jacobi method for sparse matrices.

    Args:
        A: Sparse coefficient matrix (scipy.sparse.csr_matrix).
        b: Right-hand side vector (numpy array).
        x0: Initial guess for the solution vector (numpy array).
        tol: Tolerance for convergence.
        max_iter: Maximum number of iterations.

    Returns:
        x: Approximate solution vector.
        iterations: Number of iterations performed.
        time_taken: Time taken for the iterations.
    """
    n=A.shape[0]
    x=x0.copy()
    x_new=x0.copy()
    D=A.diagonal()
    L_U=A-sparse.diags(D)
    start_time=time.time()
    for k in range(max_iter):
        x_new=(b-L_U.dot(x))/D
        error = np.linalg.norm(x_new-x)
        x = x_new
        if error < tol:
            break
    end_time = time.time()
    time_taken = end_time - start_time
    return x_new, 1, time_taken



def gauss_seidel_sparse_with_error(A, b, x0, x_exact, tol=eps, max_iter=iteration):
    ### TODO: 
    # Add comments first. 
    # Add your code
    return x0, 0, []

### TODO: 
# Set up all the important parameters
# Set up all useful plotting tools
