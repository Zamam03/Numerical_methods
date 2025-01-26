import numpy as np
import sympy as sp



def dfp_multivariable(f, vars, init_guess, tol=1e-6, max_iter=100):
    """
    DFP (Rank-Two Update) Algorithm for multivariable optimization with automatic gradient computation.

    Parameters:
    f: sympy expression, the objective function
    vars: list of sympy symbols, the variables of the objective function
    init_guess: list or np.array, initial guess for the variables
    tol: float, tolerance for stopping criteria
    max_iter: int, maximum number of iterations

    Returns:
    x_min: np.array, the point where the minimum occurs
    f_min: float, the minimum value of the function
    """
    # Compute the gradient (first derivatives)
    grad_f = [sp.diff(f, var) for var in vars]
    
    # Convert sympy expressions to lambda functions for numerical evaluation
    grad_f_lambdas = [sp.lambdify(vars, grad, "numpy") for grad in grad_f]
    f_lambda = sp.lambdify(vars, f, "numpy")
    
    # Initialize the current point and the inverse Hessian approximation (identity matrix)
    x_curr = np.array(init_guess)
    n = len(x_curr)
    G = np.eye(n)

    for i in range(max_iter):
        # Evaluate the gradient at the current point
        grad_curr = np.array([grad_f_lambda(*x_curr) for grad_f_lambda in grad_f_lambdas], dtype=float)

        # Check for convergence based on gradient norm
        if np.linalg.norm(grad_curr) < tol:
            print(f'Converged in {i+1} iterations')
            break
        
        # Compute search direction
        d_k = -G @ grad_curr
        
        # Line search: use secant method to find the optimal alpha
        alpha_func = lambda alpha: f_lambda(*x_curr + alpha * d_k)
        alpha_k = secant_method_stationary_points(alpha_func, 1, 2)

        # Update current point
        x_next = x_curr + alpha_k * d_k

        # Evaluate new gradient
        grad_next = np.array([grad_f_lambda(*x_next) for grad_f_lambda in grad_f_lambdas], dtype=float)

        # Compute updates for DFP
        delta_k = alpha_k * d_k
        gamma_k = grad_next - grad_curr

        # DFP Update formula (using outer products)
        term1 = np.outer(delta_k, delta_k) / np.dot(delta_k, gamma_k)
        term2 = G @ np.outer(gamma_k, gamma_k) @ G / (gamma_k.T @ G @ gamma_k)
        G = G + term1 - term2

        # Update for next iteration
        x_curr = x_next

    else:
        print("Maximum iterations reached without convergence")
    
    # Calculate the minimum value of the function
    f_min = f_lambda(*x_curr)
    
    return x_curr, f_min




def secant_method_stationary_points(f, alpha0, alpha1, tol=1e-6, max_iter=100):
    alpha = sp.symbols('alpha')
    f_prime = sp.diff(f(alpha), alpha)
    f_prime_func = sp.lambdify(alpha, f_prime, 'numpy')
    alpha_n_minus_1 = alpha0
    alpha_n = alpha1
    for _ in range(max_iter):
        f_prime_alpha_n = f_prime_func(alpha_n)
        f_prime_alpha_n_minus_1 = f_prime_func(alpha_n_minus_1)
        if f_prime_alpha_n - f_prime_alpha_n_minus_1 == 0:
            raise ValueError("Denominator in secant formula became zero, cannot proceed.")
        alpha_n_plus_1 = alpha_n - f_prime_alpha_n * (alpha_n - alpha_n_minus_1) / (f_prime_alpha_n - f_prime_alpha_n_minus_1)
        if abs(alpha_n_plus_1 - alpha_n) < tol:
            break
        alpha_n_minus_1, alpha_n = alpha_n, alpha_n_plus_1
    return alpha_n

# Example usage
x1, x2 = sp.symbols('x1 x2')
f = 2*x1**2 + 2*x1*x2+  x2**2 + x1- x2

init_guess = [0.0, 0.0]



x_min, f_min = dfp_multivariable(f, [x1, x2], init_guess)
print(f'Minimum point: {x_min}')
print(f'Minimum value: {f_min}')

