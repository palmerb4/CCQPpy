import numpy as np

def projection(x, lower_bound, upper_bound):
    return np.clip(x, lower_bound, upper_bound)

def grad_f(x):
    # Define the gradient of the objective function here
    # Example: For a simple quadratic function, f(x) = x^2
    return x ** 2

# Pospisil Algorithm

def PBBf(x0, lower_bound, upper_bound, max_iter=1000, tol=1e-6, fallback=True):
    x = x0
    alpha = 1e-4
    x_old = 0

    for k in range(max_iter):
        g = grad_f(x)
        g_old = g
        
        
        if k > 0:
            y = g - g_old
            s = x - x_old
            alpha = np.dot(s, y) / np.dot(y, y)

            if fallback:
                alpha_min = 1e-4
                alpha_max = 1e4
                alpha = max(min(alpha, alpha_max), alpha_min)

        x_old = x
        x = projection(x - alpha * g, lower_bound, upper_bound)

        if np.linalg.norm(x - x_old) < tol:
            break

    return x

# Yan Algorithm

def BBPGD(x0, lower_bound, upper_bound, max_iter=1000, tol=1e-6):
    return PBBf(x0, lower_bound, upper_bound, max_iter, tol, fallback=False)

# Example usage
x0 = np.array([0.0]) # Initial point
lower_bound = np.array([-10.0]) # Lower bound
upper_bound = np.array([10.0]) # Upper bound

PBBf_x = PBBf(x0, lower_bound, upper_bound)
print("With fallback:", PBBf_x)

BBPGD_x = BBPGD(x0, lower_bound, upper_bound)
print("Without fallback:", BBPGD_x)