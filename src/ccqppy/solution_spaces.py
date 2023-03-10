"""Define the projection operations for different convex feasible sets"""

# external
from abc import ABC, abstractmethod

class ProjOpBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, x):
        """Projection operation for some convex space

        Parameters
        ----------
            x : {array-like, matrix} of shape (n_unknowns x 1)
                value to project onto the feasible set

        Returns
        -------
            x_proj : {array-like, matrix} of shape (n_unknowns x 1)
                projected value of x
        """
        pass


class IdentityProjOp(ProjOpBase):
    def __init__(self):
        pass

    def __call__(self, x):
        """Projection operation for the space 
        {x in R^n}

        Parameters
        ----------
            x : {array-like, matrix} of shape (n_unknowns x 1)
                value to project onto the feasible set

        Returns
        -------
            x_proj : {array-like, matrix} of shape (n_unknowns x 1)
                projected value of x
        """
        return x


class LowerBoundProjOp(ProjOpBase):
    def __init__(self, lower_bound):
        self.lower_bound = lower_bound

    def __call__(self, x):
        """Projection operation for the space 
        {x in R^n : x >= lb} for some lower bound lb in R^n

        Parameters
        ----------
            x : {array-like, matrix} of shape (n_unknowns x 1)
                value to project onto the feasible set

        Returns
        -------
            x_proj : {array-like, matrix} of shape (n_unknowns x 1)
                projected value of x
        """
        do_proj_mask = x < self.lower_bound
        return self.lower_bound * do_proj_mask + x * (1 - do_proj_mask)


class UpperBoundProjOp(ProjOpBase):
    def __init__(self, upper_bound):
        self.upper_bound = upper_bound

    def __call__(self, x):
        """Projection operation for the space 
        {x in R^n : x <= ub} for some upper bound lb in R^n

        Parameters
        ----------
            x : {array-like, matrix} of shape (n_unknowns x 1)
                value to project onto the feasible set

        Returns
        -------
            x_proj : {array-like, matrix} of shape (n_unknowns x 1)
                projected value of x
        """
        do_proj_mask = x > self.upper_bound
        return self.upper_bound * do_proj_mask + x * (1 - do_proj_mask)


class BoxProjOp(ProjOpBase):
    def __init__(self, lower_bound, upper_bound):
        assert(upper_bound > lower_bound,
               "Upper bound must be greater than the lower bound")
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, x):
        """Projection operation for the space 
        {x in R^n : lb <= x <= ub} for some lower and upper bounds lb/ub in R^n

        Parameters
        ----------
            x : {array-like, matrix} of shape (n_unknowns x 1)
                value to project onto the feasible set

        Returns
        -------
            x_proj : {array-like, matrix} of shape (n_unknowns x 1)
                projected value of x
        """
        do_upper_proj_mask = x > self.upper_bound
        do_lower_proj_mask = x < self.lower_bound
        return self.lower_bound * do_lower_proj_mask + self.upper_bound * do_upper_proj_mask \
            + x * (1 - do_upper_proj_mask) * (1 - do_lower_proj_mask)
