"""Benchmark each solver using a set of randomly generated 
convex constrained quadratic programming problems"""

# external
import numpy as np
from scipy.stats import wishart
import matplotlib.pyplot as plt

# internal
from ccqppy import solvers
from ccqppy import solution_spaces as ss


class BenchmarkRandomCCQP:
    """Benchmark each solver using an ensemble of randomly generated CCQPs"""

    def __init__(self, solvers_to_benchmark, convex_proj_ops_to_benchmark, problem_sizes,
                 desired_residual_tol, max_matrix_vector_multiplications=np.inf):
        # store the user input
        self.solvers_to_benchmark = solvers_to_benchmark
        self.convex_proj_ops_to_benchmark = convex_proj_ops_to_benchmark
        self.problem_sizes = problem_sizes
        self.desired_residual_tol = desired_residual_tol
        self.max_matrix_vector_multiplications = max_matrix_vector_multiplications

        # initialize the internal data
        self._num_problems = problem_sizes.shape[0]
        self._problem_residual = np.zeros(self._num_problems, dtype=float)
        self._problem_converged = np.zeros(self._num_problems, dtype=bool)
        self._problem_time = np.zeros(self._num_problems, dtype=np.float32)
        self._problem_num_matrix_vector_mults = np.zeros(
            self._num_problems, dtype=int)

    def generate_random_convex_quadratic_func(self, problem_size, seed=1234):
        """
            Generate a random f(x) = x^T A x - x^T b 

            - The Hessian us sampled from the Wishart distribution of matrices of the form B B^T \
                for random square B in R^{n x n} with the identity as the base SPD matrix
            - b is then generated using b = A x for random x in R^n 

            Parameters
            ----------
                problem_size : {int}
                    number of unknowns
                seed : {int}
                    random seed to use for the randomly generated entities

            Returns
            -------
            A : {array-like, matrix} of shape (problem_size, problem_size)
                Randomly generated hessian matrix.
            b : {array-like, matrix} of shape (problem_size, 1)   
                Randomly generated element of the range space of A.         
        """

        A = wishart.rvs(problem_size, np.eye(
            problem_size), size=1, random_state=seed)
        b = A.dot(np.random.rand(problem_size))
        return (A, b)

    def run(self):
        # benchmark each solver/projection op pair for each problem size
        result_shape = [len(self.solvers_to_benchmark),
                        len(self.convex_proj_ops_to_benchmark),
                        len(self.problem_sizes)]
        self._problem_residual = np.empty(result_shape, dtype=float)
        self._problem_converged = np.empty(result_shape, dtype=bool)
        self._problem_time = np.empty(result_shape, dtype=np.float32)
        self._problem_num_matrix_vector_mults = np.empty(
            result_shape, dtype=int)
        for solver_id, solver in enumerate(self.solvers_to_benchmark):
            for proj_id_id, convex_proj_op in enumerate(self.convex_proj_ops_to_benchmark):
                for problem_id, problem_size in enumerate(self.problem_sizes):
                    # initialize the solver and the projection op
                    # provide the bare minimum arguments, let everything else be default
                    solver_instance = solver(
                        self.desired_residual_tol, self.max_matrix_vector_multiplications)
                    convex_proj_op_instance = convex_proj_op(problem_size)

                    # generate and run the problem
                    A, b = self.generate_random_convex_quadratic_func(
                        problem_size, problem_id)
                    result = solver_instance.solve(
                        A, b, convex_proj_op=convex_proj_op_instance)

                    # store the results
                    self._problem_residual[solver_id, proj_id_id,
                                           problem_id] = result.solution_residual
                    self._problem_converged[solver_id, proj_id_id,
                                            problem_id] = result.solution_converged
                    self._problem_time[solver_id, proj_id_id,
                                       problem_id] = result.solution_time
                    self._problem_num_matrix_vector_mults[solver_id, proj_id_id,
                                                          problem_id] = result.solution_num_matrix_vector_multiplications

    def process_results(self):
        num_solvers = len(self.solvers_to_benchmark)
        num_proj_ops = len(self.convex_proj_ops_to_benchmark)

        # Plot the number of matrix vector multiplications and compute time for all cases vs the matrix size.
        fig, axs_left = plt.subplots(num_solvers, num_proj_ops, sharex='col', sharey='row',
                                     gridspec_kw={'hspace': 0, 'wspace': 0})  # , constrained_layout = True

        axs_left = axs_left.reshape([num_solvers, num_proj_ops])
        axs_right = np.copy(axs_left)
        for i in range(num_solvers):
            for j in range(num_proj_ops):
                axs_right[i, j] = axs_left[i, j].twinx()
                axs_left[i, j].get_shared_y_axes().join(
                    axs_left[i, j], axs_right[i, j])

        for solver_id, solver in enumerate(self.solvers_to_benchmark):
            for proj_id_id, convex_proj_op in enumerate(self.convex_proj_ops_to_benchmark):
                axs_left[solver_id, proj_id_id].semilogy(
                    self.problem_sizes, self._problem_time[solver_id, proj_id_id, :], 'b')
                axs_right[solver_id, proj_id_id].semilogy(
                    self.problem_sizes, self._problem_num_matrix_vector_mults[solver_id, proj_id_id, :], 'r')

                axs_left[solver_id, proj_id_id].yaxis.label.set_color('red')
                axs_right[solver_id, proj_id_id].yaxis.label.set_color('blue')
                axs_left[solver_id, proj_id_id].tick_params(axis='y', colors='red')
                axs_right[solver_id, proj_id_id].tick_params(axis='y', colors='blue')
                axs_left[solver_id, proj_id_id].label_outer()
                axs_right[solver_id, proj_id_id].label_outer()


                fake_solver_instance = solver(0)
                fake_convex_proj_op_instance = convex_proj_op(0)
                if solver_id == 0:
                    axs_left[solver_id, proj_id_id].set_title(fake_convex_proj_op_instance.name)

                if solver_id == num_solvers - 1:
                    axs_left[solver_id, proj_id_id].set_xlabel('number of unknowns')


                if proj_id_id == 0:
                    axs_left[solver_id, proj_id_id].set_ylabel(fake_solver_instance.name + '\n wall-clock time [s]')

                if proj_id_id == num_proj_ops - 1:
                    axs_right[solver_id, proj_id_id].set_ylabel(fake_solver_instance.name + '\n number of matrix-vector multiplications')
                    
        plt.show()
        print("")

if __name__ == '__main__':
    problem_sizes = np.linspace(2, 500, 50, dtype=int)
    desired_residual_tol = 1e-5
    max_matrix_vector_multiplications = 5000

    solvers_to_benchmark = [solvers.CCQPSolverAPGD, solvers.CCQPSolverAPGDAntiRelaxation]
    convex_proj_ops_to_benchmark = [ss.IdentityProjOp,
                                    ss.LowerBoundProjOp,
                                    ss.UpperBoundProjOp,
                                    ss.BoxProjOp,
                                    ss.SphereProjOp,
                                    ss.ConeProjOp]

    benchmark = BenchmarkRandomCCQP(solvers_to_benchmark, convex_proj_ops_to_benchmark,
                                    problem_sizes, desired_residual_tol, max_matrix_vector_multiplications)
    benchmark.run()
    benchmark.process_results()
