"""This file contains a suite of analytical test problems to test against"""

from abc import ABC, abstractmethod, abstractproperty
import numpy as np

# internal
from ccqppy import solution_spaces as ss


class TestProblemBase(ABC):
    """Abstract base class for generating analytical constrained quadratic programming test problems.

    Because non-unique solutions are hard to compare against, these problems are restructed to 
    those with unique solutions. 
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractproperty
    def number_of_unknowns(self):
        pass

    @abstractproperty
    def A(self):
        pass

    @abstractproperty
    def b(self):
        pass

    @abstractproperty
    def convex_proj_op(self):
        pass

    @abstractproperty
    def exact_solution(self):
        pass


class UnconstrainedSPD1(TestProblemBase):
    """A simple 3D unconstrained, strictly convex QP problem"""

    def __init__(self):
        pass

    @property
    def number_of_unknowns(self):
        return 3

    @property
    def A(self):
        return np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])

    @property
    def b(self):
        return -self.A.dot(self.exact_solution)

    @property
    def convex_proj_op(self):
        return ss.IdentityProjOp(3)

    @property
    def exact_solution(self):
        return np.array([1, 0, 1])


class UnconstrainedSPD2(TestProblemBase):
    """A simple 3D unconstrained, strictly convex QP problem"""

    def __init__(self):
        pass

    @property
    def number_of_unknowns(self):
        return 3

    @property
    def A(self):
        return np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])

    @property
    def b(self):
        return -self.A.dot(self.exact_solution)

    @property
    def convex_proj_op(self):
        return ss.DisjointProjOp(ss.IdentityProjOp(1), ss.IdentityProjOp(1), ss.IdentityProjOp(1))

    @property
    def exact_solution(self):
        return np.array([1, 0, 1])


class BoxConstrainedSPD(TestProblemBase):
    """A simple 3D constrained, strictly convex QP problem 
    with unconstrained solution falling within the valid domain"""

    def __init__(self):
        pass

    @property
    def number_of_unknowns(self):
        return 3

    @property
    def A(self):
        return np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])

    @property
    def b(self):
        return -self.A.dot(self.exact_solution)

    @property
    def convex_proj_op(self):
        return ss.BoxProjOp(3, lower_bound=np.array([0, 0, 0]), upper_bound=np.array([2, 2, 2]))

    @property
    def exact_solution(self):
        return np.array([1, 0, 1])


class ThinBoxConstrainedSPD(TestProblemBase):
    """A simple 3D constrained, strictly convex QP problem 
    with unconstrained solution falling within the valid (albeit inconvenient) domain"""

    def __init__(self):
        pass

    @property
    def number_of_unknowns(self):
        return 3

    @property
    def A(self):
        return np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])

    @property
    def b(self):
        return -self.A.dot(self.exact_solution)

    @property
    def convex_proj_op(self):
        return ss.BoxProjOp(3, lower_bound=np.array([-10, -0.1, 0.9]), upper_bound=np.array([10, 0.1, 1.1]))

    @property
    def exact_solution(self):
        return np.array([1, 0, 1])


class ActiveBoxConstrainedSPD(TestProblemBase):
    """A simple 3D constrained, strictly convex QP problem 
    with unconstrained solution falling outside the valid domain"""

    def __init__(self):
        pass

    @property
    def number_of_unknowns(self):
        return 3

    @property
    def A(self):
        return np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])

    @property
    def b(self):
        return -self.A.dot(np.array([1, 1, 1]))

    @property
    def convex_proj_op(self):
        return ss.BoxProjOp(3, lower_bound=np.array([9, 9, 9]), upper_bound=np.array([10, 10, 10]))

    @property
    def exact_solution(self):
        return np.array([9, 9, 9])
