import numpy as np
import matplotlib.pyplot as plt

def generate_linear_system(n):
  """
  Generates a linear system with a diagonally dominant matrix A and vector b.

  Args:
    n: Dimension of the system (size of the matrix A).

  Returns:
    A: Coefficient matrix (numpy array).
    b: Right-hand side vector (numpy array).
  """

  A = -1 * np.ones((n, n))
  for i in range(n):
    A[i,i]  = 5*(i+1)
    
  b = np.random.rand(n)

  return A, b

# Example usage:
n = 100  # Dimension of the system
A, b = generate_linear_system(n)

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

def jacobi_method(A, b, x0, tol=1e-5, max_iter=1000):
  """
  Implements the Jacobi method for solving the linear system Ax = b.

  Args:
    A: Coefficient matrix (numpy array).
    b: Right-hand side vector (numpy array).
    x0: Initial guess for the solution vector (numpy array).
    tol: Tolerance for convergence.
    max_iter: Maximum number of iterations.

  Returns:
    x: Approximate solution vector.
    iterations: Number of iterations performed.
    errors: List of errors between exact and approximate solution at each iteration.
  """
  ### TODO: Review code here

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

  return x, k + 1, errors


def plot_error(errors, iterations):
  plt.figure(figsize=(8, 6))
  plt.plot(range(iterations), errors, marker='o', linestyle='-')
  plt.semilogy(range(iterations), errors, marker='o', linestyle='-')  # Use semilogy for log-scale on y-axis
  plt.xlabel("Iterations")
  plt.ylabel("Error estimate")
  plt.title("Error vs Iterations (Jacobi Method)")
  plt.grid(True)
  plt.show()


# Example usage:
n = 5
A, b = generate_linear_system(n)  # Generate a linear system
x0 = np.zeros(np.size(b))
print(A)
# Solve using Jacobi method
x_jacobi, iterations, errors = jacobi_method(A, b, x0)

# Calculate exact solution
x_exact = np.linalg.solve(A, b)

# Print results
print(f"Iterations: {iterations}")
print(f"Solution Jacobi: {x_jacobi}")
print(f"Exact solution: {x_exact}")

# Plot the error
plot_error(errors, iterations)