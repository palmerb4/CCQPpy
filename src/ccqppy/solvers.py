# external
from abc import ABC, abstractmethod, abstractproperty
import numpy as np
import time

# internal
from . import solution_spaces as ss


class CCQPSolverBase(ABC):
    """Abstract base class for constrained quadratic programming problems."""

    @abstractmethod
    def __init__(self, desired_residual_tol, max_matrix_vector_multipliciations=np.inf):
        pass

    @abstractmethod
    def solve(self, A, b, x0=None, convex_proj_op=None):
        """
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
                projection operator taking x to its projection onto the feasible set.

        Returns
        -------
        self : CCQPSolverBase
            The solved constrained convex quadratic problem.
        """
        pass

    def _checkSolveInput(self, A, b, x0):
        pass

    @abstractproperty
    def name(self):
        """Return the name of the solver."""
        pass

    @abstractproperty
    def solution(self):
        pass

    @abstractproperty
    def solution_residual(self):
        pass

    @abstractproperty
    def solution_converged(self):
        pass

    @abstractproperty
    def solution_time(self):
        pass

    @abstractproperty
    def solution_num_matrix_vector_multiplications(self):
        pass


class CCQPSolverAPGD(CCQPSolverBase):
    """Concrete implementation of the APGD algorithm

    Parameters
    ----------
    desired_residual_tol : numerical_type or None.
        desired residual to accept the iterative solution.
    max_matrix_vector_multiplications : numerical_type or None. Defaults to infinity.
        Maximum number of matrix-vector multiplies before the solver is terminated early.
    """

    def __init__(self, desired_residual_tol, max_matrix_vector_multiplications=np.inf, with_anti_relaxation=False):
        # store the user input
        self.desired_residual_tol = desired_residual_tol
        self.max_matrix_vector_multiplications = max_matrix_vector_multiplications
        self.with_anti_relaxation = with_anti_relaxation

        # initialize the internal data
        self._solution = None
        self._solution_residual = None
        self._solution_converged = None
        self._solution_time = None
        self._solution_num_matrix_vector_mults = None

    def solve(self, A, b, x0=None, convex_proj_op=None):
        """
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
        self : CCQPSolverAPGD
            The solved constrained convex quadratic problem.
        """
        num_unknowns = b.shape[0]
        if convex_proj_op is None:
            convex_proj_op = ss.IdentityProjOp(num_unknowns)

        if self.with_anti_relaxation:
            result = self._solve_method1(A, b, x0, convex_proj_op)
        else:
            result = self._solve_method0(A, b, x0, convex_proj_op)
        return result

    def _solve_method0(self, A, b, x0=None, convex_proj_op=None):
        """APDG optimized for QP from Algorithm 6 of Pospisil 2015"""
        num_unknowns = b.shape[0]
        if convex_proj_op is None:
            convex_proj_op = ss.IdentityProjOp(num_unknowns)

        time_start = time.time()
        self._checkSolveInput(A, b, x0)

        print("solving APGD")
        mv_count = 0

        # set the initial guess if not given
        if x0 is None:
            x0 = np.zeros(num_unknowns)

        # line 1 to 3 of Pospisil 2015
        xk = np.copy(x0)
        yk = np.copy(x0)
        thetak = 1.0
        thetakp1 = 1.0

        # line 4 of Pospisil 2015
        xkdiff = xk - np.ones(num_unknowns)
        AxkdiffNorm2 = np.linalg.norm(A.dot(xkdiff))
        xkdiffNorm2 = np.linalg.norm(xkdiff)
        mv_count += 1

        Lk = AxkdiffNorm2 / xkdiffNorm2
        tk = 1.0 / Lk

        # enter main loop
        while True:
            # line 7 of Mazhar 2015
            # Axb = A yk, this does not change in the following Lipchitz loop
            Ayk = A.dot(yk)
            mv_count += 1
            if mv_count >= self.max_matrix_vector_multiplications:
                break

            gk = Ayk + b

            # line 5 of Pospisil 2015
            xkp1 = convex_proj_op(yk - tk * gk)

            rightTerm1 = yk.dot(Ayk) * 0.5
            rightTerm2 = yk.dot(b)

            while True:
                # calc Lipchitz condition
                Axkp1 = A.dot(xkp1)
                mv_count += 1
                if mv_count >= self.max_matrix_vector_multiplications:
                    break

                # line 9 of Mazhar 2015
                leftTerm1 = xkp1.dot(Axkp1) * 0.5
                leftTerm2 = xkp1.dot(b)

                xkdiff = xkp1 - yk
                rightTerm3 = gk.dot(xkdiff)
                rightTerm4 = 0.5 * Lk * xkdiff.dot(xkdiff)
                if (leftTerm1 + leftTerm2) <= (rightTerm1 + rightTerm2 + rightTerm3 + rightTerm4):
                    break

                # line 10 & 11 of Mazhar 2015
                Lk *= 2
                tk = 1.0 / Lk

                # line 12 of Mazhar 2015
                xkp1 = convex_proj_op(yk - tk * gk)

            # line7 and 8 of Pospisil 2015
            thetakp1 = 0.5 * (-thetak * thetak + thetak *
                              np.sqrt(4 + thetak * thetak))
            betakp1 = thetak * (1 - thetak) / (thetak * thetak + thetakp1)
            ykp1 = (1 + betakp1) * xkp1 - betakp1 * xk

            # check convergence, line 4 of Pospisil 2015
            # res = np.linalg.norm(
            #     Lk * (xkp1 - convex_proj_op(xkp1 - tk * (Axkp1 + b))))
            gd = 1e-6
            res = np.linalg.norm(1.0 / (3 * num_unknowns * gd) *
                                 (xkp1 - convex_proj_op(xkp1 - gd * (Axkp1 + b))))
            if res < self.desired_residual_tol:
                break

            # next iteration
            Lk *= 0.9
            tk = 1.0 / Lk

            # swap the contents of pointers directly, be careful
            yk, ykp1 = np.frombuffer(ykp1), np.frombuffer(yk)
            xk, xkp1 = np.frombuffer(xkp1), np.frombuffer(xk)
            thetak = thetakp1

        self._solution = np.copy(xkp1)
        self._solution_converged = mv_count < self.max_matrix_vector_multiplications
        self._solution_residual = res
        self._solution_num_matrix_vector_mults = mv_count
        time_stop = time.time()
        self._solution_time = time_stop - time_start

        return self

    def _solve_method1(self, A, b, x0=None, convex_proj_op=None):
        """APDG with anti-relaxation from Mazhar 2015"""
        num_unknowns = b.shape[0]
        if convex_proj_op is None:
            convex_proj_op = ss.IdentityProjOp(num_unknowns)

        time_start = time.time()
        self._checkSolveInput(A, b, x0)

        print("solving APGD")
        mv_count = 0

        # set the initial guess if not given
        if x0 is None:
            x0 = np.zeros(num_unknowns)

        # line 1 and 2 of Mazhar 2015
        xk = np.copy(x0)
        yk = np.copy(x0)
        xhatk = np.ones(num_unknowns)

        # line 3 of Mazhar 2015
        thetak = 1.0
        thetakp1 = 1.0

        # line 4 and 5 of Mazhar 2015
        xkdiff = xk - xhatk
        AxkdiffNorm2 = np.linalg.norm(A.dot(xkdiff))
        xkdiffNorm2 = np.linalg.norm(xkdiff)
        mv_count += 1

        Lk = AxkdiffNorm2 / xkdiffNorm2
        tk = 1.0 / Lk

        # enter main loop
        resmin = np.inf
        while True:
            # line 7 of Mazhar 2015
            # Axb = A yk, this does not change in the following Lipchitz loop
            Ayk = A.dot(yk)
            mv_count += 1
            if mv_count >= self.max_matrix_vector_multiplications:
                break

            gk = Ayk + b

            # line 8 of Mazhar 2015
            xkp1 = convex_proj_op(yk - tk * gk)

            rightTerm1 = yk.dot(Ayk) * 0.5
            rightTerm2 = yk.dot(b)

            while True:
                # calc Lipchitz condition
                Axkp1 = A.dot(xkp1)
                mv_count += 1
                if mv_count >= self.max_matrix_vector_multiplications:
                    break

                # line 9 of Mazhar 2015
                leftTerm1 = xkp1.dot(Axkp1) * 0.5
                leftTerm2 = xkp1.dot(b)

                xkdiff = xkp1 - yk
                rightTerm3 = gk.dot(xkdiff)
                rightTerm4 = 0.5 * Lk * xkdiff.dot(xkdiff)
                if (leftTerm1 + leftTerm2) <= (rightTerm1 + rightTerm2 + rightTerm3 + rightTerm4):
                    break

                # line 10 & 11 of Mazhar 2015
                Lk *= 2
                tk = 1.0 / Lk

                # line 12 of Mazhar 2015
                xkp1 = convex_proj_op(yk - tk * gk)

            # line 14-16 of Mazhar 2015
            thetakp1 = 0.5 * (-thetak * thetak + thetak *
                              np.sqrt(4 + thetak * thetak))
            betakp1 = thetak * (1 - thetak) / (thetak * thetak + thetakp1)
            ykp1 = (1 + betakp1) * xkp1 - betakp1 * xk

            # check convergence, line 17 and Eq 25 of Mazhar 2015
            gd = 1e-6
            res = np.linalg.norm(1.0 / (3 * num_unknowns * gd) *
                                 (xkp1 - convex_proj_op(xkp1 - gd * (Axkp1 + b))))

            # line 18-21 of Mazhar 2015
            if res < resmin:
                resmin = res
                xhatk = np.copy(xkp1)

            # line 22-24 of Mazhar 2015
            if res < self.desired_residual_tol:
                break

            # line 25-28 of Mazhar 2015
            if gk.dot(xkp1 - xk) > 0:
                ykp1 = np.copy(xkp1)
                thetakp1 = 1

            # line 29-30 of Mazhar 2015
            Lk *= 0.9
            tk = 1.0 / Lk

            # next iteration
            # swap the contents of pointers directly, be careful
            yk, ykp1 = np.frombuffer(ykp1), np.frombuffer(yk)
            xk, xkp1 = np.frombuffer(xkp1), np.frombuffer(xk)
            thetak = thetakp1

        # line 32 of Maxhar 2015
        self._solution = np.copy(xhatk)

        self._solution_converged = mv_count < self.max_matrix_vector_multiplications
        self._solution_residual = res
        self._solution_num_matrix_vector_mults = mv_count
        time_stop = time.time()
        self._solution_time = time_stop - time_start

        return self

    @property
    def name(self):
        return "APGD"

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
