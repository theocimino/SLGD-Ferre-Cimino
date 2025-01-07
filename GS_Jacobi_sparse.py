import numpy as np
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import time

# GLOBAL VARIABLES:
eps = 1e-5
iteration = 10000

# FUNCTIONS
def generate_corrected_sparse_tridiagonal_matrix(n, x0, diagonal_value=5, off_diagonal_value=1):
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
    main_diag = np.full(n, diagonal_value + off_diagonal_value)

    # Construct sparse matrix
    # Construction of data of non-zero scalars
    data = np.zeros(3 * n - 2)
    data = np.concatenate(([diagonal_value,off_diagonal_value], (n-2)*[off_diagonal_value , diagonal_value , off_diagonal_value], [off_diagonal_value, diagonal_value]))
    # Construction of rows and columns vector used to build the good-looking matrix (tri diagonal)
    rows = [0, 0]
    cols = np.arange(2)
    for i in range(1, n - 1):
        rows = np.concatenate((rows, [i, i, i]))
        cols = np.concatenate((cols, np.arange(i - 1, i + 2)))
    cols = np.concatenate((cols, [n - 2, n - 1]))
    rows = np.concatenate((rows, [n - 1, n - 1]))
    # Creation of the sparse matrix with the scipy built-in function
    As = csr_matrix((data, (rows, cols)), shape=(n, n))

    # Creation of an arbitrary b vector as the right-side of the equation
    b = np.zeros_like(x0)
    return As, b

def generate_sparse_tridiagonal_matrix(n,x0):
    """
    Generates a sparse tridiagonal matrix with the specific values.

    Args:
        n: Dimension of the system (size of the matrix A).

    Returns:
        A: Sparse coefficient matrix (scipy.sparse.csr_matrix).
        b: Right-hand side vector (numpy array).
    """

    h=1/(n+1)
    print(h)
    h_off=-1/(h**2)
    return generate_corrected_sparse_tridiagonal_matrix(n,x0,2/(h**2),h_off)


def jacobi_sparse_with_error(A, b, x0, x_exact, tol=eps, max_iter=iteration):
    """
    Jacobi method for sparse matrices with error calculation and debugging.

    Args:
        A: Sparse coefficient matrix (scipy.sparse.csr_matrix).
        b: Right-hand side vector (numpy array).
        x0: Initial guess for the solution vector (numpy array).
        x_exact: Exact solution vector for error calculation (numpy array).
        tol: Tolerance for convergence.
        max_iter: Maximum number of iterations.

    Returns:
        x: Approximate solution vector.
        iterations: Number of iterations performed.
        time_taken: Time taken for the iterations.
        errors: List of errors at each iteration.
    """
    n = A.shape[0]
    x = x0.copy()
    D = A.diagonal()
    if np.any(D == 0):
        raise ValueError("Matrix A has zero on the diagonal, Jacobi method cannot proceed.")
    L_U = A - sparse.diags(D)
    errors = []
    start_time = time.time()
    for k in range(max_iter):
        x_new = (b - L_U.dot(x)) / D
        error = np.linalg.norm(x_new - x)
        errors.append(np.linalg.norm(x_new - x_exact))
        print(f"Iteration {k}: x_new = {x_new}, error = {error}")
        print(f"D = {D}, L_U = {L_U.toarray()}, b = {b}, x = {x}")
        x = x_new
        if error < tol:
            break
    end_time = time.time()
    time_taken = end_time - start_time
    return x, k + 1, time_taken, errors

def gauss_seidel_sparse_with_error(A, b, x0, x_exact, tol=eps, max_iter=iteration):
    """
    Gauss-Seidel method for sparse matrices with error calculation and debugging.

    Args:
        A: Sparse coefficient matrix (scipy.sparse.csr_matrix).
        b: Right-hand side vector (numpy array).
        x0: Initial guess for the solution vector (numpy array).
        x_exact: Exact solution vector for error calculation (numpy array).
        tol: Tolerance for convergence.
        max_iter: Maximum number of iterations.

    Returns:
        x: Approximate solution vector.
        iterations: Number of iterations performed.
        time_taken: Time taken for the iterations.
        errors: List of errors at each iteration.
    """
    n = A.shape[0]
    x = x0.copy()
    D = A.diagonal()
    L = sparse.tril(A, -1)
    L = L.tocsr()
    U = sparse.triu(A, 1)
    U = U.tocsr()
    errors = []
    start_time = time.time()
    for k in range(max_iter):
        for i in range(n):
            sum1 = L[i, :].dot(x)
            sum2 = U[i, :].dot(x)
            x[i] = (b[i] - sum1 - sum2) / D[i]
        error = np.linalg.norm(x - x_exact)
        errors.append(error)
        print(f"Iteration {k}: x = {x}, error = {error}")
        if error < tol:
            break
    end_time = time.time()
    time_taken = end_time - start_time
    return x, k + 1, time_taken, errors



def test_rayon(A):
    eigvalues, eigvectors = np.linalg.eig(A.toarray())
    print(eigvalues)
    #print(a)
    rayon=np.linalg.norm(np.linalg.eigvals(A.toarray()), np.inf)
    if rayon>1:
        return False, rayon
    return True, rayon


# Parameters
n = 50
x0 = np.random.rand(n) # Initial guess

# Generate matrices and vectors
A_sparse, b = generate_sparse_tridiagonal_matrix(n,x0)
x_exact = np.linalg.solve(A_sparse.toarray(), b)
#print(f"A_sparse {A_sparse.toarray()}")
verif_rayon, valeur_rayon=test_rayon(A_sparse)

# Jacobi method
x_jacobi, iter_jacobi, time_jacobi, errors_jacobi = jacobi_sparse_with_error(A_sparse, b, x0, x_exact)
#print(f"errors_jacobi : {errors_jacobi} et x_jacobi : {x_jacobi}")

# Gauss-Seidel method
x_gs, iter_gs, time_gs, errors_gs = gauss_seidel_sparse_with_error(A_sparse, b, x0, x_exact)
#print(f"errors_gs : {errors_gs} et x_gs : {x_gs}")

# Plotting the errors
plt.plot(errors_jacobi, label='Jacobi')
plt.plot(errors_gs, label='Gauss-Seidel')
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.legend()
plt.show()

print(f"Jacobi method: Iterations = {iter_jacobi}, Time = {time_jacobi:.4f} seconds")
print(f"Gauss-Seidel method: Iterations = {iter_gs}, Time = {time_gs:.4f} seconds")
print(f"test de la condition sur le rayon spectrale : {verif_rayon} \n rayon spectrale : {valeur_rayon}")


