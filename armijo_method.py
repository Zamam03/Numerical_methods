def arjo_method(x, alpha=0.1, scaling_factor=0.5):
    for i in range(10):
        grad = grad_f(x)
        x = x - alpha * scaling_factor * grad
        print(f"Iteration {i+1}: x = {x}, f(x) = {f(x)}")
    return x

x = np.array([2.0,2.0])
arjo_method(x)