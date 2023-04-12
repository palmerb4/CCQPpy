"""This file contains a suite of analytical test problems to test against

Each te

"""

from abc import ABC, abstractmethod, abstractproperty


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
