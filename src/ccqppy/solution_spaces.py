"""Define the projection operations for different convex feasible sets"""

# external
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt


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

    def plot(self, embedded_dimension, num_random_samples, lower_bound, upper_bound):
        """Plot the projection operator applied to s random samples taken from the box 
        {x in R^n : lb <= x <= ub} for some lower and upper bounds lb/ub in R^n

        Parameters
        ----------
            embedded_dimension : {int}
                dimention of the embeded space
            num_random_samples : {int} 
                number of random samples
        """
        assert(np.all(upper_bound > lower_bound),
               "Upper bound must be greater than the lower bound")
        assert(embedded_dimension <= 3,
               "Visualizing high dimensional spaces is not supported. Possible values are [1,2,3]")

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for _ in range(num_random_samples):
            random_x_in_box = np.random.rand(
                embedded_dimension) * (upper_bound - lower_bound) + lower_bound
            proj_x = self.__call__(random_x_in_box)
            ax.scatter(*proj_x)
        plt.show()


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
        assert(np.all(upper_bound > lower_bound),
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


class SphereProjOp(ProjOpBase):
    def __init__(self, radius):
        self.radius = radius

    def __call__(self, x):
        """Projection operation for the space 
        {x in R^n : sqrt(x^Tx) <= r} for some hyper-sphere radius r in R^+

        Parameters
        ----------
            x : {array-like, matrix} of shape (n_unknowns x 1)
                value to project onto the feasible set

        Returns
        -------
            x_proj : {array-like, matrix} of shape (n_unknowns x 1)
                projected value of x
        """
        x_norm = np.linalg.norm(x)
        if x_norm > self.radius:
            return self.radius * x / x_norm
        else:
            return x


class ConeProjOp(ProjOpBase):
    def __init__(self, aspect_ratio):
        assert(aspect_ratio > 0, "Aspect ratio must be positive")
        self.aspect_ratio = aspect_ratio

    def __call__(self, x):
        """Projection operation for the space 
        {[x,z]^T in R^{n-1} x R : x^Tx <= mu^2 z^2} for some hyper-cone aspect ratio mu in R^+

        Parameters
        ----------
            x : {array-like, matrix} of shape (n_unknowns x 1)
                value to project onto the feasible set

        Returns
        -------
            x_proj : {array-like, matrix} of shape (n_unknowns x 1)
                projected value of x
        """
        x_norm = np.linalg.norm(x)

        if self.aspect_ratio * x[-1] >= x_norm:
            return x
        elif -x[-1] / self.aspect_ratio >= x_norm:
            return np.zeros_like(x)
        else:
            return (x[-1] + self.aspect_ratio * x_norm) / (self.aspect_ratio**2 + 1) \
                * np.concatenate((x[:-1] / x_norm, [-self.aspect_ratio]))
