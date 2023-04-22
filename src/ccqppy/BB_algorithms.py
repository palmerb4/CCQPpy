import numpy as np

class CCQPSolverBBPGDf(CCQPSolverBase):
    """ Concrete implementation of the BBPGD with fallback algorithm
    
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
        """BBPGD with fallback from Algorithm 5 of Pospisil 2015b
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
        self : CCQPSolverBBPGDf
            The solved constrained convex quadratic problem.
        """
        num_unknowns = b.shape[0]
        if convex_proj_op is None:
            convex_proj_op = ss.IdentityProjOp(num_unknowns)
        
        time_start = time.time()
        self._checkSolveInput(A, b, x0)
        
        print("solving BBPGDf")
        mv_count = 0

        # set the initial guess if not given
        if x0 is None:
            x0 = np.zeros(num_unknowns)
            
        # lines 1 and 2 of Pospisil 2015
        alpha = 1e-4
        xk = np.copy(x0)
        xkm1 = np.copy(x0)
        xkp1 = np.copy(x0)
        xmin = np.copy(x0)
        khat = 0
        K = 5 # Don't know what to set variable to

        mv_count += 1

        # enter main loop
        while True:
            # line 4 of Pospisil 2015
            s = xk - xkm1
            alphabb = s.T.dot(s)/s.T.dot(A).dot(s)
            
            gk = xk.dot(A) + b
            
            # line 5 of Pospisil 2015
            xkp1 = convex_proj_op(xk-alphabb*gk)
            
            # fallback update from Pospisil 2015
            fxkp1 = xkp1.T.dot(A).dot(xkp1)*0.5 - xkp1.T.dot(b)
            fxmin = xmin.T.dot(A).dot(xmin)*0.5 - xmin.T.dot(b)
            
            if fxkp1 < fxmin:
                gkp1 = xkp1.dot(A) + b
                
                xmin = convex_proj_op(xkp1-alpha*gkp1)
                khat = x0
                
            else:
                khat += 1
                
            # fallback application from Pospisil 2015
            if khat >= K:
                gmin = xmin.dot(A) + b
                
                xkp1 = convex_proj_op(xmin-alpha*gmin)
                xmin = xkp1
                khat = 0
                
            # check convergence, line 3 of Pospisil 2015
            res = np.linalg.norm() # I don't know what goes here, NEEDS UPDATE
            if res < self.desired_residual_tol:
                break                
            
            mv_count += 1
            xkm1 = xk
            xk = xkp1     

        self._solution = np.copy(xkp1)
        self._solution_converged = mv_count < self.max_matrix_vector_multiplications
        self._solution_residual = res
        self._solution_num_matrix_vector_mults = mv_count
        time_stop = time.time()
        self._solution_time = time_stop - time_start

        return self
    
    @property
    def name(self):
        return "BBPDGf"

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
    
class CCQPSolverBBGPD(CCQPSolverBase):
    """ Concrete implementation of the BBPGD algorithm
    
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
        """BBPGD from Algorithm 1 of Yan 2019
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
        self : CCQPSolverBBPGD
            The solved constrained convex quadratic problem.
        """

        num_unknowns = b.shape[0]
        if convex_proj_op is None:
            convex_proj_op = ss.IdentityProjOp(num_unknowns)
         
        time_start = time.time()
        self._checkSolveInput(A, b, x0)
         
        print("solving BBPGDf")
        mv_count = 0
        
        # set the initial guess if not given
        if x0 is None:
            x0 = np.zeros(num_unknowns)
             
        # initial variables
        xk = np.copy(x0)
        xkm1 = np.copy(x0)
        
        # lines 1 to 4 of Yan 2019
        g0 = A.dot(x0) + b # fn below is incorrect, NEEDS UPDATE
        if g0 >= self.desired_residual_tol: # skips algorithm if soln is x0
            alphakm1 = g0.T.dot(g0)/(g0.T.dot(A).dot(g0))
            alphak = np.copy(alphakm1)
            gkm1 = np.copy(g0)
            while True:
                xk = xkm1 - alphakm1.dot(gkm1) # descent step
                # projection step goes here - unclear notation, NEEDS UPDATE
                gk = A.dot(xk) + b # compute gradient
                res = 0 # unclear on contents of fn, NEEDS UPDATE
                # check convergence, line 11 of Yan 2019
                if res < self.desired_residual_tol:
                    break
                
                # update variables for iteration
                skm1 = xk - xkm1
                xkm1 = gk - gkm1
                alphakm1 = alphak
                alphak = skm1.T.dot(skm1)/(skm1.T.dot(xkm1))
                
                if mv_count <= self.max_matrix_vector_multiplications:
                    break
        
        self._solution = np.copy(xk)
        self._solution_converged = mv_count < self.max_matrix_vector_multiplications
        self._solution_residual = res
        self._solution_num_matrix_vector_mults = mv_count
        time_stop = time.time()
        self._solution_time = time_stop - time_start     
             
    @property
    def name(self):
        return "BBGPD"
    
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
