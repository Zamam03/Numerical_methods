import numpy as np

#objective funcion and its gradient
def f(x):
    return x[0]**2 + x[1]**2

def grad_f(x):
    return np.array([2*x[0],  2*x[1]])

x = np.array([2.0, 2.0])

#the steepest descent method
alpha = 0.1 #is our stepsize
for i in range(10): 
    x = x - alpha*grad_f(x)
    print(f"Iteration {i+1}: x = {x}, f(x) = {f(x)}")