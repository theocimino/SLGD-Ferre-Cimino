import numpy as np
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
import time


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
    ### TODO: Complete code here 
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
    return As, A_dense, b

#function used to calculate the next iteration x in the jacobi_dense function
def x_next(A,b,x_old):
  n = A.shape[0]
  x_next = np.zeros_like(x_old)
  for i in range(n):
    sum = 0
    for j in range(n):
      if(j!=i):
        sum += A[i][j] * x_old[j]
    x_next[i] = 1/(A[i,i]) * (b[i]-sum)
  return x_next


def jacobi_dense(A, b, x0, tol=1e-6, max_iter=1000):
    """
    Jacobi method for dense matrices.

    Args:
        A: Dense coefficient matrix (numpy array).
        b: Right-hand side vector (numpy array).
        x0: Initial guess for the solution vector (numpy array).
        tol: Tolerance for convergence.
        max_iter: Maximum number of iterations.

    Returns:
        x: Approximate solution vector.
        iterations: Number of iterations performed.
        time_taken: Time taken for the iterations.
    """
    start_time = time.time()
    x_new = x0.copy()
    n = A.shape[0]
    x = x0.copy()
    errors = []
    for k in range(max_iter):
        x_new = x_next(A,b,x)
        error = np.linalg.norm(x_new - x)
        errors.append(error)
        print(error)
        x = x_new
        if error < tol:
            break
    end_time = time.time()
    time_taken = end_time - start_time
    return x, k+1, time_taken



def jacobi_sparse(A, b, x0, tol=1e-7, max_iter=10000):
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




n=100
x0 = np.zeros(n)  ## initial guess
A_sparse, A_dense_v1, b = generate_corrected_sparse_tridiagonal_matrix(n) 
A_dense_v2 = A_sparse.toarray()  # Convert to dense format for comparison





# Classical Jacobi (dense)
x_dense, iter_dense, time_dense = jacobi_dense(A_dense_v2, b, x0) 
print(x_dense)
# Jacobi for sparse matrix
x_sparse, iter_sparse, time_sparse = jacobi_sparse(A_sparse, b, x0)


print(f"Iterations (dense): {iter_dense}, Time (dense): {time_dense:.4f} seconds")
print(f"Iterations (sparse): {iter_sparse}, Time (sparse): {time_sparse:.4f} seconds")

x_exact = np.linalg.solve(A_dense_v2, b)
print(x_exact)
print(x_sparse)

# ### TODO: 
# # Implement a small for loop comparing the times required for both approaches as a function of the dimension n
tab_val=[10,100,150,200,300,400,500]
for i in range(0,len(tab_val)):
    
print(generate_corrected_sparse_tridiagonal_matrix(3, diagonal_value=5, off_diagonal_value=1))
