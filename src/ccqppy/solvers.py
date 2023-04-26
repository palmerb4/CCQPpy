# external
from abc import ABC, abstractmethod, abstractproperty
import numpy as np
import time
from collections import deque

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


class CCQPSolverPGD(CCQPSolverBase):
    """Concrete implementation of PGD algorithm
    Parameters
    ----------
    desired_residual_tol : numerical_type or None.
        desired residual to accept the iterative solution.
    max_matrix_vector_multiplications : numerical_type or None. Defaults to infinity.
        Maximum number of matrix-vector multiplies before the solver is terminated early.
    """

    def __init__(self, desired_residual_tol, max_matrix_vector_multiplications=np.inf, step_size=0.01):
        # store the user input
        self.desired_residual_tol = desired_residual_tol
        self.max_matrix_vector_multiplications = max_matrix_vector_multiplications
        self.step_size = step_size

        # initialize the internal data
        self._solution = None
        self._solution_residual = None
        self._solution_converged = None
        self._solution_time = None
        self._solution_num_matrix_vector_mults = None

    def solve(self, A, b, x0=None, convex_proj_op=None):
        """Baseline PGD
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
        self : CCQPSolverPGD
            The solved constrained convex quadratic problem.
        """
        num_unknowns = b.shape[0]
        if convex_proj_op is None:
            convex_proj_op = ss.IdentityProjOp(num_unknowns)

        time_start = time.time()
        self._checkSolveInput(A, b, x0)

        print("solving PGD")
        mv_count = 0

        # set the initial guess if not given
        if x0 is None:
            x0 = np.zeros(num_unknowns)

        # initial variables
        xk = np.copy(x0)
        xkm1 = np.copy(x0)

        # lines 1 to 4 of Yan 2019
        gkm1 = A.dot(xkm1) + b
        mv_count += 1

        # check convergence, line 17 and Eq 25 of Mazhar 2015
        gd = 1e-6
        res = np.linalg.norm(1.0 / (3 * num_unknowns * gd) *
                             (xkm1 - convex_proj_op(xkm1 - gd * gkm1)))

        # skip the algorithm if the initial guess is correct.
        if res >= self.desired_residual_tol:
            while True:
                # perform the gradient descent step
                xk = convex_proj_op(xkm1 - self.step_size * gkm1)

                # get the new gradient
                gk = A.dot(xk) + b
                mv_count += 1
                if mv_count >= self.max_matrix_vector_multiplications:
                    break

                # check for convergence, line 17 and Eq 25 of Mazhar 2015
                res = np.linalg.norm(1.0 / (3 * num_unknowns * gd) *
                                     (xk - convex_proj_op(xk - gd * gk)))
                if res < self.desired_residual_tol:
                    break

                # swap the contents of pointers directly, be careful
                xk, xkm1 = np.frombuffer(xkm1), np.frombuffer(xk)
                gk, gkm1 = np.frombuffer(gkm1), np.frombuffer(gk)

        self._solution = np.copy(xk)
        self._solution_converged = mv_count < self.max_matrix_vector_multiplications
        self._solution_residual = res
        self._solution_num_matrix_vector_mults = mv_count
        time_stop = time.time()
        self._solution_time = time_stop - time_start

        return self

    @property
    def name(self):
        return "PGD"

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


class CCQPSolverAPGD(CCQPSolverBase):
    """Concrete implementation of the APGD algorithm

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
        """APDG from Algorithm 6 of Pospisil 2015
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


class CCQPSolverAPGDAntiRelaxation(CCQPSolverBase):
    """Concrete implementation of the APGD algorithm

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
        """APDG with anti-relaxation from Mazhar 2015
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
        return "Anti-relaxation APGD"

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


class CCQPSolverBBPGD(CCQPSolverBase):
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
        gkm1 = A.dot(xkm1) + b
        mv_count += 1

        # check convergence, line 17 and Eq 25 of Mazhar 2015
        gd = 1e-6
        res = np.linalg.norm(1.0 / (3 * num_unknowns * gd) *
                             (xkm1 - convex_proj_op(xkm1 - gd * gkm1)))

        # skip the algorithm if the initial guess is correct.
        if res >= self.desired_residual_tol:
            alpha = gkm1.dot(gkm1) / (gkm1.dot(A.dot(gkm1)))
            while True:
                # perform the gradient descent step
                xk = convex_proj_op(xkm1 - alpha * gkm1)

                # get the new gradient
                gk = A.dot(xk) + b
                mv_count += 1
                if mv_count >= self.max_matrix_vector_multiplications:
                    break

                # check for convergence, line 17 and Eq 25 of Mazhar 2015
                res = np.linalg.norm(1.0 / (3 * num_unknowns * gd) *
                                     (xk - convex_proj_op(xk - gd * gk)))
                if res < self.desired_residual_tol:
                    break

                # update variables for iteration
                xkdiff = xk - xkm1
                gkdiff = gk - gkm1
                alpha = xkdiff.dot(
                    xkdiff) / (xkdiff.dot(gkdiff) + 10 * np.finfo(float).eps)

                # swap the contents of pointers directly, be careful
                xk, xkm1 = np.frombuffer(xkm1), np.frombuffer(xk)
                gk, gkm1 = np.frombuffer(gkm1), np.frombuffer(gk)

        self._solution = np.copy(xk)
        self._solution_converged = mv_count < self.max_matrix_vector_multiplications
        self._solution_residual = res
        self._solution_num_matrix_vector_mults = mv_count
        time_stop = time.time()
        self._solution_time = time_stop - time_start

        return self

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

        # initial variables
        # lines 1 and 2 of Pospisil 2015
        xk = np.copy(x0)
        xkm1 = np.copy(x0)

        xmin = np.copy(x0)
        gmin = np.copy(x0)

        # lines 1 to 4 of Yan 2019
        gkm1 = A.dot(xkm1) + b
        mv_count += 1

        # check convergence, line 17 and Eq 25 of Mazhar 2015
        gd = 1e-6
        res = np.linalg.norm(1.0 / (3 * num_unknowns * gd) *
                             (xkm1 - convex_proj_op(xkm1 - gd * gkm1)))

        resmin = np.inf
        # skip the algorithm if the initial guess is correct.
        if res >= self.desired_residual_tol:
            alpha = gkm1.dot(gkm1) / (gkm1.dot(A.dot(gkm1)))
            while True:
                # perform the gradient descent step
                xk = convex_proj_op(xkm1 - alpha * gkm1)

                # get the new gradient
                gk = A.dot(xk) + b
                mv_count += 1
                if mv_count >= self.max_matrix_vector_multiplications:
                    break

                # check for convergence, line 17 and Eq 25 of Mazhar 2015
                res = np.linalg.norm(1.0 / (3 * num_unknowns * gd) *
                                     (xk - convex_proj_op(xk - gd * gk)))
                if res < self.desired_residual_tol:
                    break

                # fallback
                if res < resmin:
                    resmin = np.copy(res)
                    xmin = np.copy(xk)
                    gmin = np.copy(gk)

                # apply fallback upon stagnation
                if alpha < 10 * np.finfo(float).eps:
                    xk = convex_proj_op(xmin - gd * gmin)

                # update variables for iteration
                xkdiff = xk - xkm1
                gkdiff = gk - gkm1
                alpha = xkdiff.dot(
                    xkdiff) / (xkdiff.dot(gkdiff) + 10 * np.finfo(float).eps)

                # swap the contents of pointers directly, be careful
                xk, xkm1 = np.frombuffer(xkm1), np.frombuffer(xk)
                gk, gkm1 = np.frombuffer(gkm1), np.frombuffer(gk)

        self._solution = np.copy(xk)
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
                 m=5, tau=0.5, sigma1=0.01, sigma2=0.5):
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
        gk = A.dot(xk) + b
        fk = np.dot(gk, xk)
        alpha = gk.dot(gk) / (gk.dot(A.dot(gk)))
        mv_count += 2

        tau = self.t
        m = self.m
        sig1 = self.sigma1
        sig2 = self.sigma2
        fk_queue = deque(maxlen=m)
        fk_queue.append(fk)

        # enter main loop
        while True:
            # alpha is the step size
            dk = convex_proj_op(xk - alpha * gk) - xk
            Adk = A.dot(dk)
            mv_count += 1
            if mv_count >= self.max_matrix_vector_multiplications:
                break

            # Precompute the dot products, line 7 of Popisil 2018
            dkdotdk = np.dot(dk, dk)
            dkdotAdk = np.dot(dk, Adk)
            dkdotgk = np.dot(dk, gk)

            # Breaking conditions
            if np.sqrt(dkdotdk) <= self.desired_residual_tol:
                break

            # line 9 of popisil 2018
            fmax = max(fk_queue)

            # lines 10-18 of popisil 2018
            xi = (fmax - fk) / dkdotAdk
            beta = -dkdotgk / dkdotAdk
            betahat = tau * beta + np.sqrt((tau ** 2) * (beta ** 2) + 2 * xi)
            betak = np.random.uniform(low=sig1, high=min(betahat, sig2))

            xk += betak * dk
            gk += betak * Adk
            fk += betak * betak * dkdotgk + 0.5 * (betak ** 2) * dkdotAdk
            fk_queue.append(fk)

            alpha = dkdotdk / dkdotAdk

        self._solution = np.copy(xk)
        self._solution_converged = mv_count < self.max_matrix_vector_multiplications
        self._solution_residual = np.sqrt(dkdotdk)
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


class CCQPSolverMPRGP(CCQPSolverBase):
    """Concrete implementation of the MPRGP algorithm
    from Alg 5.8 of OPTIMAL QUADRATIC PROGRAMMING ALGORITHMS

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
        """MPRGP
        f(x) = 1/2 x^T A x - x^T b

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

        time_start = time.time()
        self._checkSolveInput(A, b, x0)

        print("solving MPRGP")
        mv_count = 0

        # set the initial guess if not given
        if x0 is None:
            x0 = np.zeros(num_unknowns)

        # Step 0: Initialization
        xk = convex_proj_op(x0)
        xkp1 = np.copy(xk)
        gk = A.dot(xk) + b
        gkp1 = np.copy(gk)
        mv_count += 1

        # check convergence, line 17 and Eq 25 of Mazhar 2015
        gd = 1e-6
        res = np.linalg.norm(1.0 / (3 * num_unknowns * gd) *
                             (xk - convex_proj_op(xk - gd * gk)))

        # skip the algorithm if the initial guess is correct.
        if res >= self.desired_residual_tol:
            # compute the initial BB step size
            alpha_bb = gk.dot(gk) / (gk.dot(A.dot(gk)))
            mv_count += 1

            # Line 4 of algorith 5.8
            delta_xk = np.isclose(xk, convex_proj_op(xk))
            p = delta_xk * gk

            while True:
                # Precomputations
                Axk = A.dot(xk)
                mv_count += 1
                if mv_count >= self.max_matrix_vector_multiplications:
                    break
                gk = Axk + b

                # Prepotioning condition from line 6 of algorithm 5.8
                delta_xk = np.isclose(xk, convex_proj_op(xk))
                psi_xk = delta_xk * gk
                n_xk = convex_proj_op.normal_vector(xk)
                beta_xk = (1 - delta_xk) * \
                    (gk - np.min([0, n_xk.dot(gk)]) * n_xk)
                if beta_xk.dot(beta_xk) < psi_xk.dot(psi_xk):
                    # Precomputations
                    Ap = A.dot(p)
                    mv_count += 1
                    if mv_count >= self.max_matrix_vector_multiplications:
                        break

                    # line 8 of algorithm 5.8
                    alpha_cg = psi_xk.dot(p) / p.dot(Ap)
                    y = xk - alpha_cg * p

                    # alternative to line 9 of algorithm 5.8
                    # here, we use recursive bisection to determine a step size
                    # within the feasible solution space.
                    alpha_f = alpha_cg + 10 * np.finfo(float).eps
                    while True:
                        yf = xk - alpha_f * p
                        if np.all(np.isclose(yf, convex_proj_op(yf))):
                            break
                        else:
                            alpha_f *= 0.5

                    # line 10-12 of algorithm 5.8
                    if alpha_cg <= alpha_f:
                        # conjugate gradient step
                        # line 11 of algorithm 5.8
                        xkp1 = np.copy(y)
                        gkp1 = gk - alpha_cg * Ap

                        xkdiff = xkp1 - xk
                        gkdiff = gkp1 - gk
                        alpha_bb = xkdiff.dot(xkdiff) / (xkdiff.dot(A.dot(xkdiff)) + 10 * np.finfo(float).eps)

                        # line 12 of algorithm 5.8
                        delta_y = np.isclose(y, convex_proj_op(y))
                        psi_y = delta_y * gkp1
                        beta = psi_y * Ap / p.dot(Ap)
                        p = psi_y - beta * p
                    else:
                        # extension step using BB step size
                        # line 15. note, there is a typo. g = g - alphaf Ap. means
                        # g^{k+1/2} = g^k - alphaf Ap.
                        xkphalf = xk - alpha_f * p
                        gkphalf = gk - alpha_f * Ap

                        # line 16 with BB step
                        xkdiff = xkphalf - xk
                        gkdiff = gkphalf - gk
                        alpha = xkdiff.dot(
                            xkdiff) / (xkdiff.dot(gkdiff) + 10 * np.finfo(float).eps)
                        xkp1 = convex_proj_op(xkphalf - alpha * gkphalf)

                        # reset the GC algorithm
                        # line 17
                        gkp1 = A.dot(xkp1) + b
                        mv_count += 1
                        if mv_count >= self.max_matrix_vector_multiplications:
                            break

                        delta_xkp1 = np.isclose(xkp1, convex_proj_op(xkp1))
                        psi_xkp1 = delta_xkp1 * gkp1
                        p = np.copy(psi_xkp1)

                        xkdiff = xkp1 - xk
                        gkdiff = gkp1 - gk
                        alpha_bb = xkdiff.dot(xkdiff) / (xkdiff.dot(A.dot(xkdiff)) + 10 * np.finfo(float).eps)
                else:
                    # proportioning step from line 20
                    # line 21 but with BB step
                    xkp1 = convex_proj_op(xk - alpha_bb * gk)

                    xkdiff = xkp1 - xk
                    gkdiff = gkp1 - gk
                    alpha_bb = xkdiff.dot(
                        xkdiff) / (xkdiff.dot(A.dot(xkdiff)) + 10 * np.finfo(float).eps)

                    gk = A.dot(xk) + b
                    mv_count += 1
                    if mv_count >= self.max_matrix_vector_multiplications:
                        break

                    # reset the CG iteraton
                    delta_xkp1 = np.isclose(xkp1, convex_proj_op(xkp1))
                    psi_xkp1 = delta_xkp1 * gkp1
                    p = np.copy(psi_xkp1)

                res = np.linalg.norm(1.0 / (3 * num_unknowns * gd) *
                                     (xkp1 - convex_proj_op(xkp1 - gd * gkp1)))
                if res < self.desired_residual_tol:
                    break

                # swap the contents of pointers directly, be careful
                xk, xkp1 = np.frombuffer(xkp1), np.frombuffer(xk)
                gk, gkp1 = np.frombuffer(gkp1), np.frombuffer(gk)

        self._solution = np.copy(xkp1)
        self._solution_converged = mv_count < self.max_matrix_vector_multiplications
        self._solution_residual = res
        self._solution_num_matrix_vector_mults = mv_count
        time_stop = time.time()
        self._solution_time = time_stop - time_start

        return self

    @property
    def name(self):
        return "MPRGP"

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
