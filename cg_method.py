from scipy.optimize import minimize

def cg_method():
    result = minimize(f, x0=[2.0, 2.0], method='CG', jac=grad_f)
    print(f"Optimal point: {resut.x}, f(x) = {result.fun}")

cg_method()