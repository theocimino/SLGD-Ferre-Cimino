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
    #print(h)
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
        #print(f"Iteration {k}: x_new = {x_new}, error = {error}")
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
        #print(f"Iteration {k}: x = {x}, error = {error}")
        if error < tol:
            break
    end_time = time.time()
    time_taken = end_time - start_time
    return x, k + 1, time_taken, errors



def rayon_spectral_jacobi(A):
    B=A.toarray()
    D=np.diag(np.diag(B))
    L=np.tril(B,-1)
    U=np.triu(B, 1)
    T_jacobi=np.linalg.inv(D).dot(L+U)
    return np.linalg.norm(np.linalg.eigvals(T_jacobi),np.inf)

def rayon_spectral_gs(A):
    B = A.toarray()
    L = np.tril(B, -1)
    U = np.triu(B, 1)
    D = np.diag(np.diag(B))
    T_gs = np.linalg.inv(D + L).dot(-U)
    return np.linalg.norm(np.linalg.eigvals(T_gs), np.inf)


def rayon_spectral_sor(A,w):
    L=np.tril(A.toarray(),-1)
    D=np.diag(np.diag(A.toarray()))
    U=np.triu(A.toarray(),1)
    D_inv=np.linalg.inv(D+w*L)
    T_sor=(1-w)*np.eye(A.shape[0])+w*D_inv.dot(U)
    return np.linalg.norm(np.linalg.eigvals(T_sor),np.inf)

def rayon_spectral_ssor(A,w):
    D=np.diag(np.diag(A.toarray()))
    L=np.tril(A.toarray(),-1)
    U= np.triu(A.toarray(), 1)
    D_w_L_inv=np.linalg.inv(D+w*L)
    D_w_U_inv=np.linalg.inv(D+w*U)
    T_ssor=D_w_L_inv.dot((1-w)*D).dot(D_w_U_inv)
    return np.linalg.norm(np.linalg.eigvals(T_ssor),np.inf)




def test_rayon(A,w=1.5):
    eigvalues, eigvectors = np.linalg.eig(A.toarray())
    #print(eigvalues)
    rayon_methodes=[rayon_spectral_jacobi(A),rayon_spectral_gs(A),rayon_spectral_sor(A,w),rayon_spectral_ssor(A,w)]
    verif_rayon=[]
    valeur_rayon=[]
    for i in range(0,len(rayon_methodes)):
        if rayon_methodes[i]>1:
            verif_rayon.append(False)
            valeur_rayon.append(rayon_methodes[i])
        else :
            verif_rayon.append(True)
            valeur_rayon.append(rayon_methodes[i])
    return verif_rayon,valeur_rayon


def sor_method(A, b, x0, x_exact, valeur_rayon, tol=eps, max_iter=iteration):
    #w=2/(1+np.sqrt(1-valeur_rayon**2))
    w=1.5
    n = A.shape[0]
    x = x0.copy()
    s = x0.copy()
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
            s[i] = (b[i] - sum1 - sum2) / D[i]
            x[i] = w*s[i]+(1-w)*x[i]
        error = np.linalg.norm(x - x_exact)
        errors.append(error)
        #print(f"Iteration {k}: x = {x}, error = {error}")
        if error < tol:
            break
    end_time = time.time()
    time_taken = end_time - start_time
    return x, k + 1, time_taken, errors, w



def ssor_method(A, b, x0, x_exact, valeur_rayon, tol=eps, max_iter=iteration):
    """
    SSOR (Symmetric Successive Over-Relaxation) method for sparse matrices with error calculation and debugging.

    Args:
        A: Sparse coefficient matrix (scipy.sparse.csr_matrix).
        b: Right-hand side vector (numpy array).
        x0: Initial guess for the solution vector (numpy array).
        x_exact: Exact solution vector for error calculation (numpy array).
        tol: Tolerance for convergence.
        max_iter: Maximum number of iterations.
        w: Relaxation factor.

    Returns:
        x: Approximate solution vector.
        iterations: Number of iterations performed.
        time_taken: Time taken for the iterations.
        errors: List of errors at each iteration.
    """
    #w=2/(1+np.sqrt(1-valeur_rayon**2))
    w=1.5
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
        # Forward sweep
        for i in range(n):
            sum1 = L[i, :].dot(x)
            sum2 = U[i, :].dot(x)
            x[i] = (1 - w) * x[i] + (w / D[i]) * (b[i] - sum1 - sum2)
        
        # Backward sweep
        for i in range(n-1, -1, -1):
            sum1 = L[i, :].dot(x)
            sum2 = U[i, :].dot(x)
            x[i] = (1 - w) * x[i] + (w / D[i]) * (b[i] - sum1 - sum2)
        
        error = np.linalg.norm(x - x_exact)
        errors.append(error)
        #print(f"Iteration {k}: x = {x}, error = {error}")
        if error < tol:
            break
            
    end_time = time.time()
    time_taken = end_time - start_time
    return x, k + 1, time_taken, errors





#évolution du rayon spectrale et de w en fonction de n


def test_methods(tab_valeurs):
    start=time.time()
    omega_list=[]
    temps_jacobi=[]
    temps_gs=[]
    temps_sor=[]
    temps_ssor=[]
    iterations_jacobi=[]
    iterations_gs=[]
    iterations_sor=[]
    iterations_ssor=[]
    rayon_jacobi=[]
    rayon_gs=[]
    rayon_sor=[]
    rayon_ssor=[]
    for i in range(0,len(tab_valeurs)):
        print(f"Execution du code en dimension : {tab_valeurs[i]}")
        n=tab_valeurs[i]
        x0 = np.random.rand(n) # Initial guess

        # Generate matrices and vectors
        A_sparse, b = generate_sparse_tridiagonal_matrix(n,x0)
        x_exact = np.linalg.solve(A_sparse.toarray(), b)
        #print(f"A_sparse {A_sparse.todense()}")
        verif_rayon, valeur_rayon=test_rayon(A_sparse)


        # Jacobi method
        x_jacobi, iter_jacobi, time_jacobi, errors_jacobi = jacobi_sparse_with_error(A_sparse, b, x0, x_exact)
        #print(f"errors_jacobi : {errors_jacobi} et x_jacobi : {x_jacobi}")

        # Gauss-Seidel method
        x_gs, iter_gs, time_gs, errors_gs = gauss_seidel_sparse_with_error(A_sparse, b, x0, x_exact)
        #print(f"errors_gs : {errors_gs} et x_gs : {x_gs}")

        # Sor method
        x_sor, iter_sor, time_sor, errors_sor, w = sor_method(A_sparse, b, x0, x_exact, valeur_rayon)
        #print(f"errors_sor : {errors_sor} et x_sor : {x_sor}")

        # Add SSOR method to the methods comparison
        x_ssor, iter_ssor, time_ssor, errors_ssor = ssor_method(A_sparse, b, x0, x_exact, valeur_rayon)


        #liste des coefficients de relaxation
        omega_list.append(w)

        #liste des temps respectifs
        temps_jacobi.append(time_jacobi)
        temps_gs.append(time_gs)
        temps_sor.append(time_sor)
        temps_ssor.append(time_ssor)

        #liste des itérations respectives
        iterations_jacobi.append(iter_jacobi)
        iterations_gs.append(iter_gs)
        iterations_sor.append(iter_sor)
        iterations_ssor.append(iter_ssor)

        #liste des rayons spectraux de chaque méthodes
        rayon_jacobi.append(valeur_rayon[0])
        rayon_gs.append(valeur_rayon[1])
        rayon_sor.append(valeur_rayon[2])
        rayon_ssor.append(valeur_rayon[3])

    
    
    end=time.time()
    tps=end-start
    print(f"Temps d'éxecution du programme {tps}\n")


    """
    # Plotting the errors compare with iteration
    plt.figure("comparaison de l'erreur en fonction des itérations") 
    plt.plot(errors_jacobi, label='Jacobi')
    plt.plot(errors_gs, label='Gauss-Seidel')
    plt.plot(errors_sor, label='Sor')
    plt.plot(errors_ssor, label='Ssor')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

    print(f"test de la condition sur le rayon spectral : \n- Pour la méthode de Jacobi : {verif_rayon[0]}\n- Pour la méthode de Gauss-Seidel : {verif_rayon[1]}\n- Pour la méthode SOR : {verif_rayon[2]}\n- Pour la méthode SSOR : {verif_rayon[3]}\n \nRayon spectral : \n-Pour la méthode de Jacobi : {rayon_jacobi[0]}\n-Pour la méthode de Gauss-Seidel : {rayon_gs[0]}\n-Pour la méthode SOR : {rayon_sor[0]}\n-Pour la méthode SSOR : {rayon_sor[0]}\n")
    print(f"Jacobi method: Iterations = {iter_jacobi}, Time = {time_jacobi:.4f} seconds")
    print(f"Gauss-Seidel method: Iterations = {iter_gs}, Time = {time_gs:.4f} seconds")
    print(f"Sor method: Iterations = {iter_sor}, Time = {time_sor:.4f} seconds")
    print(f"SSOR method: Iterations = {iter_ssor}, Time = {time_ssor:.4f} seconds")
    """


    '''
    # Plotting omega and spectral_radius
    plt.figure("Rayon Spectral et Coefficient de Relaxation en fonction de la dimension")
    plt.plot(tab_valeurs,omega_list,label='Omega')
    plt.plot(tab_valeurs, rayon_list, label='Rayon')
    plt.xlabel('Dimension')
    plt.ylabel('Valeur')
    plt.legend()
    plt.show()
    '''

    '''
    # Plotting spectral radius in fonction of relaxion coefficient
    plt.figure("Rayon Spectral en fonction du coefficient de relaxation")
    plt.plot(omega_list,rayon_list)
    plt.xlabel('Omega')
    plt.ylabel('Rayon Spectral')
    plt.title (" Rayon spectral en fonction du paramètre de relaxation")
    plt.show()
    '''

    '''
    #Methods Time comparison
    plt.figure("Comparaison des Methods")
    plt.subplot(2,1,1)
    plt.plot(tab_valeurs, temps_jacobi, label='Temps Jacobi')
    plt.plot(tab_valeurs, temps_gs, label='Temps Gauss-Seidel')
    plt.plot(tab_valeurs, temps_sor, label='Temps Sor')
    plt.plot(tab_valeurs, temps_ssor, label='Temps Ssor')
    plt.xlabel('Dimensions')
    plt.ylabel('Temps')
    plt.legend()
    plt.grid()


    #Methods Iterations comparison
    plt.subplot(2,1,2)
    plt.plot(tab_valeurs, iterations_jacobi, label='Itérations Jacobi')
    plt.plot(tab_valeurs, iterations_gs, label='Itérations  Gauss-Seidel')
    plt.plot(tab_valeurs, iterations_sor, label='Itérations  Sor')
    plt.plot(tab_valeurs, iterations_ssor, label='Itérations  Ssor')
    plt.xlabel('Dimensions')
    plt.ylabel('Iterations')
    plt.legend()
    plt.grid()

    plt.tight_layout() 
    '''

    plt.figure("Rayon Spectral en fonction de la méthode")
    plt.plot(tab_valeurs, rayon_jacobi, label='Rayon Jacobi')
    plt.plot(tab_valeurs, rayon_gs, label='Rayon  Gauss-Seidel')
    plt.plot(tab_valeurs, rayon_sor, label='Rayon Sor')
    plt.plot(tab_valeurs, rayon_ssor, label='Rayon  Ssor')
    plt.xlabel('Dimension')
    plt.ylabel('Rayon')
    plt.legend()
    plt.grid()


    plt.show()


#tab_valeurs=[i for i in range(2,100,2)]
#[i*2 for i in range(2,30)]
test_methods([i for i in range(2,20)])

"""

jacobi le plus d'itération mais rapide
GS moins d'itération que jacobi environ 2x moins mais très long
Sor très rapide et très peu d'itération
SSOR le meilleur

"""
