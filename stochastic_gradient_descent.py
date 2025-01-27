def gradient_descent(x, alpha=0.1, iterations=10):
    for i in range(iterations):
        x = x - alpha * grad_f(x)
        print(f"Iterations {i+1}: x = {x}, f(x) = {f(x)}")
    return x

x = np.array([2.0, 2.0])
gradient_descent(x)