#!/usr/bin/env python
# coding: utf-8

import numpy as np


class CCQPSolverSPG(CCQPSolverBase):
    """Concrete implementation of the SPG-QP algorithm
    Parameters
    ----------
    desired_residual_tol : numerical_type or None.
        desired residual to accept the iterative solution.
    max_matrix_vector_multiplications : numerical_type or None. Defaults to infinity.
        Maximum number of matrix-vector multiplies before the solver is terminated early.
    """

    def __init__(self, desired_residual_tol, max_matrix_vector_multiplications=np.inf,
                m=5, tau=0.5, sigma1=0.01, sigma2=0.5, alpha0=1.0):
        # store the user input
        self.desired_residual_tol = desired_residual_tol
        self.max_matrix_vector_multiplications = max_matrix_vector_multiplications

        # initialize the internal data
        self._solution = None
        self._solution_residual = None
        self._solution_converged = None
        self._solution_time = None
        self._solution_num_matrix_vector_mults = None
        
        # New inputs
        self.m = m
        self.t = tau
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.alpha0 = alpha0
    
    def cost_func(A, b, x):
        return 0.5 * np.dot(x, np.dot(A, x)) - np.dot(b, x)
        
    def solve(self, A, b, x0=None, convex_proj_op=None):
            """SPG-QP from Algorithm 5 of Pospisil 2018
            f(x) = x^T A x - x^T b 
            Parameters
            ----------
                A : {array-like, matrix} of shape (n_unknowns, n_unknowns)
                    Hessian matrix of f(x).
                b : {array-like, matrix} of shape (n_unknowns, 1)
                    Element of the range space of A.
                k : {integer} of shape (n)
                    Number of iterations for this algorithm to compute.
                m : {integer} of shape(n) 
                    Number of previous iterations used to compute max value of objective function.
                    Large m creates more stable convergence, though it is computationally heavy.
                t : {integer} of shape (n) between 0 and 1.
                    Amount of safeguarding, used to scale step size.
                    Small tau means small step size and vice versa.
                x0 : {array-like, matrix} of shape (n_unknowns, 1)
                    Initial guess for the solution x. Defaults to all zeros.
                convex_proj_op : {func(x)} taking array-like x of shape (n_unknowns, 1) \
                    to its projection x_proj also of shape (n_unknowns, 1). Defaults to IdentityProjOp.
                projection operator taking x to its projection 
                    onto the feasible set.
            Returns
            -------
            self : CCQPSolverSPG
                The solved constrained convex quadratic problem.
            """
            num_unknowns = b.shape[0]
            if convex_proj_op is None:
                convex_proj_op = ss.IdentityProjOp(num_unknowns)

            time_start = time.time()
            self._checkSolveInput(A, b, x0)

            print("solving SPG")
            mv_count = 0

            # set the initial guess if not given
            if x0 is None:
                x0 = np.zeros(num_unknowns)

            # line 1 to 3 of Pospisil 2018
            xk = np.copy(x0)
            gk = A @ xk - b
            fk = 1/2 * np.dot(gk, xk)
            gd = 1e-6
            
            tau = self.t
            alpha0 = self.alpha0
            k = self.k
            m = self.m
            sig1 = self.sigma1
            sig2 = self.sigma2
            
            mv_count += 1

            # enter main loop
            while True:
                dk = convex_proj_op((xk - alpha0 * gk) - xk) # Alpha0 = initial step size
                Adk = A @ dk  # Matrix-Vector multiplication
                mv_count += 1

                # Multiple Dot Product, Line 7 of Popisil 2018
                dotdk = np.dot(dk,dk)
                dotadk = np.dot(dk, Adk)
                dotgk = np.dot(dk,gk)

                # Breaking Conditions, line 
                if np.sqrt(dotdk) <= gd:
                    break
                if mv_count >= self.max_matrix_vector_multiplications:
                    break

                # line 9 of popisil 2018
                # Grippo–Lampariello–Lucidi method (GLL)
                fmax = max([cost_func(A,b,(xk - j * dk)) for j in range(min(k, m - 1) + 1)]) ### Not too sure if this is correct
                
                # lines 10-18 of popisil 2018
                xi = (fmax - fk) / dotadk
                beta = -np.dot(gk, dk) / dotadk
                betahat = tau * beta + np.sqrt((tau ** 2) * (beta ** 2) + 2 * xi) if (tau ** 2) * (beta ** 2) + 2 * xi >= 0 else 0
                betak = np.clip(betahat, sig1, sig2)

                xk1 = xk + betak * dk
                gk1 = gk + betak * Adk
                fk = fk + betak * dotgk + (betak ** 2) / 2 * dotadk
                alphak1 = dotdk / dotadk
                k += 1
                
                # swap the contents of pointers directly
                xk, xk1 = np.frombuffer(xk1), np.frombuffer(xk)
                alpha0, alphak1 = np.frombuffer(alphak1), np.frombuffer(alpha0)
                
        self._solution = np.copy(xk1)
        self._solution_converged = mv_count < self.max_matrix_vector_multiplications
        self._solution_residual = np.linalg.norm(A @ self._solution - b)
        self._solution_num_matrix_vector_mults = mv_count
        time_stop = time.time()
        self._solution_time = time_stop - time_start

        return self        
            
            
    @property
    def name(self):
        return "SPG-QP"

    @property
    def solution(self):
        return self._solution

    @property
    def solution_residual(self):
        return self._solution_residual

    @property
    def solution_converged(self):
        return self._solution_converged

    @property
    def solution_time(self):
        return self._solution_time

    @property
    def solution_num_matrix_vector_multiplications(self):
        return self._solution_num_matrix_vector_mults