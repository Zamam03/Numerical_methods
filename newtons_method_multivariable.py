import numpy as np
import sympy as sp

def newtons_method_multivariable(f, vars, init_guess, tol=1e-6, max_iter=100):
    """
    Newton's method for multivariable optimization (minimization problem).

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
    
    # Compute the Hessian matrix (second derivatives)
    hessian_f = sp.hessian(f, vars)
    
    # Convert sympy expressions to lambda functions for numerical evaluation
    grad_f_lambdas = [sp.lambdify(vars, grad, "numpy") for grad in grad_f]
    hessian_f_lambda = sp.lambdify(vars, hessian_f, "numpy")
    
    # Initialize the current point
    x_curr = np.array(init_guess)
    
    for i in range(max_iter):
        # Evaluate the gradient and Hessian at the current point
        # Note that the "*" infront of x_curr is not a mutiplication sign, is used to unpack the vector
        grad_curr = np.array([grad_f_lambda(*x_curr) for grad_f_lambda in grad_f_lambdas], dtype=float)
        hessian_curr = np.array(hessian_f_lambda(*x_curr))
        
        # Think of updating the Hession with the new formula to be H + mu * Identity_matrix
        # where mu = max(0, epsilon - smallest eigenvalue of the hessian)
        
        
        # Check for convergence
        if np.linalg.norm(grad_curr) < tol:
            print(f'Converged in {i+1} iterations')
            break
        
        # Update the current point using Newton's update rule
        delta_x = np.linalg.solve(hessian_curr, -grad_curr)
        
        x_curr = x_curr + delta_x
        
        # Also, start thinking of incorporating the step size into your updating formula
        # x_curr = x_curr + alpha * delta
        
        
        # Check for stopping criteria based on tolerance
        if np.linalg.norm(delta_x) < tol:
            print(f'Converged in {i+1} iterations')
            break
    else:
        print("Maximum iterations reached without convergence")
           
    
    # Calculate the minimum value of the function
    f_min = sp.lambdify(vars, f, "numpy")(*x_curr)
   
    
    return x_curr, f_min

# Example usage

# Define the variables
x1, x2 = sp.symbols('x1 x2')

# Define the function (example: f(x1, x2) = x1^2 + x2^2 + 3*x1*x2)
f = x1**2 + x2**2 + 3*x1*x2

# The example below will diverge is you did not use the modified newton's Mehtod
#f= x1**4 + x1*x2 + (1+x2)**2



# Initial guess
init_guess = [0.0, 0.0]

# Call Newton's method
x_min, f_min = newtons_method_multivariable(f, [x1 , x2], init_guess)

print(f'Minimum point: {x_min}')
print(f'Minimum value: {f_min}')
