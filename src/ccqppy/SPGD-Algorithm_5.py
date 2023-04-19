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

    def __init__(self, desired_residual_tol, max_matrix_vector_multiplications=np.inf):
        # store the user input
        self.desired_residual_tol = desired_residual_tol
        self.max_matrix_vector_multiplications = max_matrix_vector_multiplications

        # initialize the internal data
        self._solution = None
        self._solution_residual = None
        self._solution_converged = None
        self._solution_time = None
        self._solution_num_matrix_vector_mults = None

    def solve(self, A, b, x0=None, convex_proj_op=None):
        """SPG-QP from Algorithm 5 of Pospisil 2018
        f(x) = x^T A x - x^T b 
        Parameters
        ----------
            A : {array-like, matrix} of shape (n_unknowns, n_unknowns)
                Hessian matrix of f(x).
            b : {array-like, matrix} of shape (n_unknowns, 1)
                Element of the range space of A.
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
        k = 0
        g0 = A @ xk - b
        fk = 1/2 * np.inner(g - b, xk)
        gd = 1e-6
        tau = self.desired_residual_tol # Not sure about this

        mv_count += 1

        # enter main loop
        while True:
            xkp1 = convex_proj_op(xk + betak * dk) ## Semi-Direct pull ##
            Axkp1 = A.dot(xkp1) ## Direct pull ##
            dk = convex_proj_op((xk - alpha0 * g0) - xk) # Alpha0 = initial step size
            Adk = A @ dk  # Matrix-Vector multiplication
            h = np.dot(dk, Adk)  # Multiple Dot-product
            
            if np.sqrt(np.inner(dk, dk)) <= gd:
                break
                
            mv_count += 1
            if mv_count >= self.max_matrix_vector_multiplications:
                break
            
            # lines 9-18 of popisil 2018
            # Grippo–Lampariello–Lucidi method (GLL)
            fmax = max([f(xk - j * dk) for j in range(min(k, m - 1)+1)]) # Need to figure out how to incorperate function f(x)
            
            xi = (fmax - f0) / np.inner(dk, Adk) # Need fmax operational
            beta = -np.inner(gk, dk) / np.inner(dk, Adk)
            betahat = tau * beta + np.sqrt((tau ** 2) * (beta ** 2) + 2 * xi) if (tau ** 2) * (beta ** 2) + 2 * xi >= 0 else 0
            
            betak = np.clip(betahat, sigma1, sigma2) # sig 1&2 lower and upper bound
            
            xk1 = xk + betak * dk
            gk1 = g0 + betak * Adk
            fk = fk + betak * np.inner(dk, g0) + (betak ** 2) / 2 * np.inner(dk, Adk)
            alphak1 = np.inner(dk, dk) / np.inner(dk, Adk)
            k += 1
            
            #Calculate Residuals
            res = np.linalg.norm(1.0 / (3 * num_unknowns * gd) *
                                 (xkp1 - convex_proj_op(xkp1 - gd * (Axkp1 + b))))
            
        self._solution = np.copy(xk)
        self._solution_converged = mv_count < self.max_matrix_vector_multiplications
        self._solution_residual = res # Confused by this portion
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





