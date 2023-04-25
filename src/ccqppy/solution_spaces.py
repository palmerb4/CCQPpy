"""Define the projection operations for different convex feasible sets"""

# external
from abc import ABC, abstractmethod, abstractproperty
import numpy as np
import matplotlib.pyplot as plt


class ProjOpBase(ABC):
    def __init__(self, embedded_dimension):
        pass

    @abstractmethod
    def __call__(self, x):
        """Projection operation for some convex space

        Parameters
        ----------
            x : {array-like, matrix} of shape (n_unknowns, 1)
                value to project onto the feasible set

        Returns
        -------
            x_proj : {array-like, matrix} of shape (n_unknowns, 1)
                projected value of x
        """
        pass

    @abstractmethod
    def normal_vector(self, x):
        """Return the unit outward normal of the projection operator at the point x on the boundary.

        If x is inside the boundary, returns the zero vector.
        If x is outside the boundary, projects x onto the boundary and then returns the normal there.
        """
        pass

    @abstractproperty
    def name(self):
        """Return the name of the projection operator."""
        pass

    @abstractproperty
    def embedded_dimension(self):
        """Return the embedded dimension of the projection operator."""
        pass

    def plot(self, num_random_samples, lower_bound, upper_bound):
        """Plot the projection operator applied to s random samples taken from the box
        {x in R^n : lb <= x <= ub} for some lower and upper bounds lb/ub in R^n

        Parameters
        ----------
            num_random_samples : {int} 
                number of random samples
        """
        assert(np.all(upper_bound > lower_bound),
               "Upper bound must be greater than the lower bound")
        assert(self.embedded_dimension <= 3,
               "Visualizing high dimensional spaces is not supported. Possible values are [1,2,3]")

        dim = self.embedded_dimension
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for _ in range(num_random_samples):
            random_x_in_box = np.random.rand(dim) * (upper_bound - lower_bound) + lower_bound
            proj_x = self.__call__(random_x_in_box)
            if dim == 1:
                ax.scatter(proj_x[0], 0.0, 0.0)
            elif dim == 2:
                ax.scatter(proj_x[0], proj_x[1], 0.0)
            elif dim == 3:
                ax.scatter(proj_x[0], proj_x[1], proj_x[2])
        plt.show()


class IdentityProjOp(ProjOpBase):
    def __init__(self, embedded_dimension):
        self.dim = embedded_dimension
        pass

    @property
    def name(self):
        """Return the name of the projection operator."""
        return "Identity"

    @property
    def embedded_dimension(self):
        """Return the embedded dimension of the projection operator."""
        return self.dim

    def normal_vector(self, x):
        """Return the unit outward normal of the projection operator at the point x on the boundary.

        If x is inside the boundary, returns the zero vector.
        If x is outside the boundary, projects x onto the boundary and then returns the normal there.
        """
        return np.zeros(self.dim)

    def projected_gradient(self, x, g):
        """Return free gradient and chopped gradient at x. Their sum is the projected gradient

        Args:
            x (np.array): location
            g (np.array): gradient

        Returns:
            g_p (tuple): (free gradient, chopped gradient)
        """

    def __call__(self, x):
        """Projection operation for the space
        {x in R^n}

        Parameters
        ----------
            x : {array-like, matrix} of shape (n_unknowns, 1)
                value to project onto the feasible set

        Returns
        -------
            x_proj : {array-like, matrix} of shape (n_unknowns, 1)
                projected value of x
        """
        return x


class LowerBoundProjOp(ProjOpBase):
    def __init__(self, embedded_dimension, lower_bound=None):
        self.dim = embedded_dimension
        if lower_bound is not None:
            self.lower_bound = lower_bound
        else:
            self.lower_bound = -np.ones(embedded_dimension)

    @property
    def name(self):
        """Return the name of the projection operator."""
        return "Lower Bound"

    @property
    def embedded_dimension(self):
        """Return the embedded dimension of the projection operator."""
        return self.dim

    def normal_vector(self, x):
        """Return the unit outward normal of the projection operator at the point x on the boundary.

        If x is inside the boundary, returns the zero vector.
        If x is outside the boundary, projects x onto the boundary and then returns the normal there.
        """
        x_proj = self.__call__(x)
        if (not np.isclose(np.linalg.norm(x - x_proj), 0)):
            return np.zeros(self.dim)
        else:
            norm = np.zeros(self.dim)
            for i in range(self.dim):
                if np.isclose(x_proj[i], self.lower_bound[i]):
                    norm[i] = -1
            return norm
        
    def projected_gradient(self,x,g):
        """Return free gradient and chopped gradient at x. Their sum is the projected gradient

        Args:
            x (np.array): location
            g (np.array): gradient

        Returns:
            g_p (tuple): (free gradient, chopped gradient)
        """
        # x_proj = self.__call__(x)
        free_g = np.zeros(self.dim)
        chopped_g = np.zeros(self.dim)
        normal = self.normal_vector(x)

        for i in range(self.dim):
            if np.isclose(x[i], self.lower_bound[i]): # In the active set
                free_g[i] = 0
                chopped_g[i] = g[i] - np.min((normal[i]*g[i],0))*(normal[i])
            else:                                     # In the free set
                free_g[i] = g[i]
                chopped_g[i] = 0
        return (free_g,chopped_g)

    def __call__(self, x):
        """Projection operation for the space
        {x in R^n : x >= lb} for some lower bound lb in R^n

        Parameters
        ----------
            x : {array-like, matrix} of shape (n_unknowns, 1)
                value to project onto the feasible set

        Returns
        -------
            x_proj : {array-like, matrix} of shape (n_unknowns, 1)
                projected value of x
        """
        do_proj_mask = x < self.lower_bound
        return self.lower_bound * do_proj_mask + x * (1 - do_proj_mask)


class UpperBoundProjOp(ProjOpBase):
    def __init__(self, embedded_dimension, upper_bound=None):
        self.dim = embedded_dimension
        if upper_bound is not None:
            self.upper_bound = upper_bound
        else:
            self.upper_bound = np.ones(embedded_dimension)

    @property
    def name(self):
        """Return the name of the projection operator."""
        return "Upper Bound"

    @property
    def embedded_dimension(self):
        """Return the embedded dimension of the projection operator."""
        return self.dim

    def normal_vector(self, x):
        """Return the unit outward normal of the projection operator at the point x on the boundary.

        If x is inside the boundary, returns the zero vector.
        If x is outside the boundary, projects x onto the boundary and then returns the normal there.
        """
        x_proj = self.__call__(x)
        if (not np.isclose(np.linalg.norm(x - x_proj), 0)):
            return np.zeros(self.dim)
        else:
            norm = np.zeros(self.dim)
            for i in range(self.dim):
                if np.isclose(x_proj[i], self.upper_bound[i]):
                    norm[i] = 1
            return norm

    def projected_gradient(self,x,g):
        """Return free gradient and chopped gradient at x. Their sum is the projected gradient

        Args:
            x (np.array): location
            g (np.array): gradient

        Returns:
            g_p (tuple): (free gradient, chopped gradient)
        """
        # x_proj = self.__call__(x)
        free_g = np.zeros(self.dim)
        chopped_g = np.zeros(self.dim)
        normal = self.normal_vector(x)

        for i in range(self.dim):
            if np.isclose(x[i], self.upper_bound[i]): # In the active set
                free_g[i] = 0
                chopped_g[i] = g[i] - np.min((normal[i]*g[i],0))*(normal[i])
            else:                                     # In the free set
                free_g[i] = g[i]
                chopped_g[i] = 0
        return (free_g,chopped_g)

    def __call__(self, x):
        """Projection operation for the space
        {x in R^n : x <= ub} for some upper bound lb in R^n

        Parameters
        ----------
            x : {array-like, matrix} of shape (n_unknowns, 1)
                value to project onto the feasible set

        Returns
        -------
            x_proj : {array-like, matrix} of shape (n_unknowns, 1)
                projected value of x
        """
        do_proj_mask = x > self.upper_bound
        return self.upper_bound * do_proj_mask + x * (1 - do_proj_mask)


class BoxProjOp(ProjOpBase):
    def __init__(self, embedded_dimension, lower_bound=None, upper_bound=None):
        self.dim = embedded_dimension
        if lower_bound is not None:
            self.lower_bound = lower_bound
        else:
            self.lower_bound = -np.ones(embedded_dimension)

        if upper_bound is not None:
            self.upper_bound = upper_bound
        else:
            self.upper_bound = np.ones(embedded_dimension)

        assert(np.all(self.upper_bound > self.lower_bound),
               "Upper bound must be greater than the lower bound.")

    @property
    def name(self):
        """Return the name of the projection operator."""
        return "Box"

    @property
    def embedded_dimension(self):
        """Return the embedded dimension of the projection operator."""
        return self.dim

    def normal_vector(self, x):
        """Return the unit outward normal of the projection operator at the point x on the boundary.

        If x is inside the boundary, returns the zero vector.
        If x is outside the boundary, projects x onto the boundary and then returns the normal there.
        """
        x_proj = self.__call__(x)
        if (not np.isclose(np.linalg.norm(x - x_proj), 0)):
            return np.zeros(self.dim)
        else:
            norm = np.zeros(self.dim)
            for i in range(self.dim):
                if np.isclose(x_proj[i], self.upper_bound[i]):
                    norm[i] = 1
                elif np.isclose(x_proj[i], self.lower_bound[i]):
                    norm[i] = -1
            return norm

    def projected_gradient(self,x,g):
        """Return free gradient and chopped gradient at x. Their sum is the projected gradient

        Args:
            x (np.array): location
            g (np.array): gradient

        Returns:
            g_p (tuple): (free gradient, chopped gradient)
        """
        free_g = np.zeros(self.dim)
        chopped_g = np.zeros(self.dim)
        normal = self.normal_vector(x)
        x_proj = self.__call__(x)

        for i in range(self.dim):
            if np.isclose(x[i], self.upper_bound[i]) or x[i]>self.upper_bound[i] \
                or np.isclose(x[i], self.lower_bound[i] or x[i]<self.upper_bound[i]): # In the active set
                free_g[i] = 0
                chopped_g[i] = g[i] - np.min((normal[i]*g[i],0))*(normal[i])
            else:                                     # In the free set
                free_g[i] = g[i]
                chopped_g[i] = 0
        return (free_g,chopped_g)

    def __call__(self, x):
        """Projection operation for the space
        {x in R^n : lb <= x <= ub} for some lower and upper bounds lb/ub in R^n

        Parameters
        ----------
            x : {array-like, matrix} of shape (n_unknowns, 1)
                value to project onto the feasible set

        Returns
        -------
            x_proj : {array-like, matrix} of shape (n_unknowns, 1)
                projected value of x
        """
        do_upper_proj_mask = x > self.upper_bound
        do_lower_proj_mask = x < self.lower_bound
        return self.lower_bound * do_lower_proj_mask + self.upper_bound * do_upper_proj_mask \
            + x * (1 - do_upper_proj_mask) * (1 - do_lower_proj_mask)


class SphereProjOp(ProjOpBase):
    def __init__(self, embedded_dimension, radius=None):
        self.dim = embedded_dimension
        if radius is not None:
            self.radius = radius
        else:
            self.radius = 1

        assert(self.radius > 0, "Radius must be positive")

    @property
    def name(self):
        """Return the name of the projection operator."""
        return "Sphere"

    @property
    def embedded_dimension(self):
        """Return the embedded dimension of the projection operator."""
        return self.dim

    def normal_vector(self, x):
        """Return the unit outward normal of the projection operator at the point x on the boundary.

        If x is inside the boundary, returns the zero vector.
        If x is outside the boundary, projects x onto the boundary and then returns the normal there.
        """
        x_proj = self.__call__(x)
        if (not np.isclose(np.linalg.norm(x - x_proj), 0)):
            return np.zeros(self.dim)
        else:
            norm = np.zeros(self.dim)
            x_proj_length = np.linalg.norm(x_proj)
            if np.isclose(x_proj_length, self.radius):
                norm = x_proj / x_proj_length
            return norm

    def projected_gradient(self,x,g):
        """Return free gradient and chopped gradient at x. Their sum is the projected gradient

        Args:
            x (np.array): location
            g (np.array): gradient

        Returns:
            g_p (tuple): (free gradient, chopped gradient)
        """
        raise NotImplementedError("Cone proximal gradient not implemented, yet.")

    def __call__(self, x):
        """Projection operation for the space
        {x in R^n : sqrt(x^Tx) <= r} for some hyper-sphere radius r in R^+

        Parameters
        ----------
            x : {array-like, matrix} of shape (n_unknowns, 1)
                value to project onto the feasible set

        Returns
        -------
            x_proj : {array-like, matrix} of shape (n_unknowns, 1)
                projected value of x
        """
        x_norm = np.linalg.norm(x)
        if x_norm > self.radius:
            return self.radius * x / x_norm
        else:
            return x


class ConeProjOp(ProjOpBase):
    # TODO(palmerb4): This projection op is bugged
    def __init__(self, embedded_dimension, aspect_ratio=None):
        self.dim = embedded_dimension
        if aspect_ratio is not None:
            self.aspect_ratio = aspect_ratio
        else:
            self.aspect_ratio = 1

        assert(self.aspect_ratio > 0, "Aspect ratio must be positive")

    @property
    def name(self):
        """Return the name of the projection operator."""
        return "Cone"

    @property
    def embedded_dimension(self):
        """Return the embedded dimension of the projection operator."""
        return self.dim

    def normal_vector(self, x):
        """Return the unit outward normal of the projection operator at the point x on the boundary.

        If x is inside the boundary, returns the zero vector.
        If x is outside the boundary, projects x onto the boundary and then returns the normal there.
        """
        raise NotImplementedError("Cone normal not implemented, yet.")

    def proximal_gradient(self,x,g):
        raise NotImplementedError("Cone proximal gradient not implemented, yet.")
    
    def __call__(self, x):
        """Projection operation for the space
        {[x,z]^T in R^{n-1} x R : x^Tx <= mu^2 z^2} for some hyper-cone aspect ratio mu in R^+

        Parameters
        ----------
            x : {array-like, matrix} of shape (n_unknowns, 1)
                value to project onto the feasible set

        Returns
        -------
            x_proj : {array-like, matrix} of shape (n_unknowns, 1)
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


class DisjointProjOp(ProjOpBase):
    def __init__(self, *convex_proj_ops):
        self.proj_ops = convex_proj_ops
        self.dim = 0
        for proj_op in self.proj_ops:
            self.dim += proj_op.embedded_dimension

    @property
    def name(self):
        """Return the name of the projection operator."""
        return "DisjointUnion"

    @property
    def embedded_dimension(self):
        """Return the embedded dimension of the projection operator."""
        return self.dim

    def normal_vector(self, x):
        """Return the unit outward normal of the projection operator at the point x on the boundary.

        If x is inside the boundary, returns the zero vector.
        If x is outside the boundary, projects x onto the boundary and then returns the normal there.
        """
        normal = np.zeros(self.dim)
        start_index = 0
        for proj_op in self.proj_ops:
            dim = proj_op.embedded_dimension
            normal[start_index: start_index +
                   dim] = proj_op.normal_vector(x[start_index: start_index + dim])
            start_index += dim
        return normal

    def projected_gradient(self,x,g):
        free_g = np.zeros(self.dim)
        chopped_g = np.zeros(self.dim)
        start_index = 0
        for proj_op in self.proj_ops:
            dim = proj_op.embedded_dimension
            f_g, c_g = proj_op.projected_gradient(x[start_index: start_index + dim],
                                                   g[start_index: start_index + dim])
            free_g[start_index: start_index+dim]= f_g
            chopped_g[start_index: start_index+dim]= c_g
            start_index += dim
        return (free_g,chopped_g)

    def __call__(self, x):
        """Projection operation for the space formed from the disjoint union of N different constraints.

        Parameters
        ----------
            x : {array-like, matrix} of shape (n_unknowns, 1)
                value to project onto the feasible set

        Returns
        -------
            x_proj : {array-like, matrix} of shape (n_unknowns, 1)
                projected value of x
        """
        x_proj = np.zeros(self.dim)
        start_index = 0
        for proj_op in self.proj_ops:
            dim = proj_op.embedded_dimension
            x_proj[start_index: start_index +
                   dim] = proj_op(x[start_index: start_index + dim])
            start_index += dim
        return x_proj
    