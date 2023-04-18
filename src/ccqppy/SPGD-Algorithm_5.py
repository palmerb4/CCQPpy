#!/usr/bin/env python
# coding: utf-8


import numpy as np

def projected_gradient_descent(f, x0, proj, A, b, m, tau, sigma1, sigma2, eps, alpha0):
    k = 0
    g = A @ x0 - b
    f_val = 1/2 * np.inner(g - b, x0)
    x = x0
    
    while True:
        d = proj(x - alpha0 * g) - x
        Ad = A @ d
        h = np.dot(d, Ad)
        
        if np.sqrt(np.inner(d, d)) <= eps:
            break
        
        fmax = max([f(x - j * d) for j in range(min(k, m - 1)+1)])
        xi = (fmax - f_val) / np.inner(d, Ad)
        beta = -np.inner(g, d) / np.inner(d, Ad)
        beta_hat = tau * beta + np.sqrt((tau ** 2) * (beta ** 2) + 2 * xi) if (tau ** 2) * (beta ** 2) + 2 * xi >= 0 else 0
        beta_k = np.clip(beta_hat, sigma1, sigma2)
        x = x + beta_k * d
        g = g + beta_k * Ad
        f_val = f_val + beta_k * np.inner(d, g) + (beta_k ** 2) / 2 * np.inner(d, Ad)
        alpha_k = np.inner(d, d) / np.inner(d, Ad)
        k += 1
    
    return x


def f(x):
    return np.dot(x, x)

# Define the projection onto the feasible set
def proj(x):
    return np.clip(x, 0, 1)

# Define the parameters
n = 10
m = 3
tau = 0.5
sigma1 = 0.1
sigma2 = 0.9
eps = 1e-6
alpha0 = 0.1

# Generate random A and b
A = np.random.randn(n, n)
b = np.random.randn(n)

# Set the initial approximation
x0 = np.random.randn(n)

# Call the function
x_opt = projected_gradient_descent(f, x0, proj, A, b, m, tau, sigma1, sigma2, eps, alpha0)

# Print the result
print("Optimal solution: ", x_opt)


plt.plot(x_opt)





