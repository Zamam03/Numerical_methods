def accelerated_gardient_descent(x, alpha=0.1, beta=0.9, iterations=10):
    v = np.zeros_like(x)
    for i in range(iterations):
        grad = grad_f(x)
        v = beta * v - alpha * grad
        x = x + v
        print(f"Iteration {i+1}: x = {x}, f(x) = {f(x)}")
    return x

x = np.array([2.0, 2.])
accelerated_gradient_descent(x)
    
    