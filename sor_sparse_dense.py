def sor_sparse(A, b, x0, w=1.5, tol=eps, max_iter=iteration):
    """
    Successive Over-Relaxation (SOR) method for sparse matrices.

    Args:
        A: Sparse coefficient matrix (scipy.sparse.csr_matrix).
        b: Right-hand side vector (numpy array).
        x0: Initial guess for the solution vector (numpy array).
        w: Relaxation factor (float).
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
    U = sparse.triu(A, 1)
    errors = []
    start_time = time.time()
    for k in range(max_iter):
        for i in range(n):
            sum1 = L[i, :].dot(x)
            sum2 = U[i, :].dot(x)
            x[i] = (1 - w) * x[i] + w * (b[i] - sum1 - sum2) / D[i]
        error = np.linalg.norm(A.dot(x) - b)
        errors.append(error)
        if error < tol:
            break
    end_time = time.time()
    time_taken = end_time - start_time
    return x, k + 1, time_taken, errors



def sor_dense(A, b, x0, w=1.5, tol=eps, max_iter=iteration):
    """
    Successive Over-Relaxation (SOR) method for dense matrices.

    Args:
        A: Dense coefficient matrix (numpy array).
        b: Right-hand side vector (numpy array).
        x0: Initial guess for the solution vector (numpy array).
        w: Relaxation factor (float).
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
    D = np.diag(A)
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    errors = []
    start_time = time.time()
    for k in range(max_iter):
        for i in range(n):
            sum1 = L[i, :].dot(x)
            sum2 = U[i, :].dot(x)
            x[i] = (1 - w) * x[i] + w * (b[i] - sum1 - sum2) / D[i]
        error = np.linalg.norm(A.dot(x) - b)
        errors.append(error)
        if error < tol:
            break
    end_time = time.time()
    time_taken = end_time - start_time
    return x, k + 1, time_taken, errors
