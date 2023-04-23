"""Benchmark each solver using a set of randomly generated 
convex constrained quadratic programming problems"""

# external
import numpy as np
from scipy.stats import wishart
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st

# internal
from ccqppy import solvers
from ccqppy import solution_spaces as ss


class BenchmarkRandomCCQP:
    """Benchmark each solver using an ensemble of randomly generated CCQPs"""

    def __init__(self, num_random_trials, solvers_to_benchmark, convex_proj_ops_to_benchmark):
        # store the user input
        self.num_trials = num_random_trials
        self.solvers_to_benchmark = solvers_to_benchmark
        self.convex_proj_ops_to_benchmark = convex_proj_ops_to_benchmark

        # initialize the internal data
        self.problem_sizes = np.zeros(
            len(convex_proj_ops_to_benchmark[0]), dtype=int)
        for i, proj_op in enumerate(self.convex_proj_ops_to_benchmark[0]):
            self.problem_sizes[i] = proj_op.embedded_dimension

        self._problem_residual = None
        self._problem_l2norm_error = None
        self._problem_converged = None
        self._problem_time = None
        self._problem_num_matrix_vector_mults = None

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
        unconstrained_solution = np.random.rand(problem_size)
        b = -A.dot(unconstrained_solution)
        return (A, b)

    def run(self):
        # benchmark each solver/projection op pair for each problem size
        result_shape = [len(self.solvers_to_benchmark),
                        len(self.convex_proj_ops_to_benchmark),
                        len(self.convex_proj_ops_to_benchmark[0]),
                        self.num_trials]
        self._problem_residual = np.zeros(result_shape, dtype=float)
        self._problem_converged = np.zeros(result_shape, dtype=int)
        self._problem_time = np.zeros(result_shape, dtype=np.float32)
        self._problem_num_matrix_vector_mults = np.zeros(
            result_shape, dtype=int)
        for solver_id, solver in enumerate(self.solvers_to_benchmark):
            for proj_type_id, proj_ops in enumerate(self.convex_proj_ops_to_benchmark):
                for proj_id, proj_op in enumerate(proj_ops):
                    for trial_id in range(self.num_trials):
                        # generate and run the problem
                        A, b = self.generate_random_convex_quadratic_func(
                            self.problem_sizes[proj_id], trial_id)
                        result = solver.solve(
                            A, b, convex_proj_op=proj_op)

                        # store the results
                        self._problem_residual[solver_id,
                                               proj_type_id,
                                               proj_id,
                                               trial_id] = result.solution_residual
                        self._problem_converged[solver_id,
                                                proj_type_id,
                                                proj_id,
                                                trial_id] = result.solution_converged
                        self._problem_time[solver_id,
                                           proj_type_id,
                                           proj_id,
                                           trial_id] = result.solution_time
                        self._problem_num_matrix_vector_mults[solver_id,
                                                              proj_type_id,
                                                              proj_id,
                                                              trial_id] = result.solution_num_matrix_vector_multiplications

    def plot(self, name, date):
        num_solvers = len(self.solvers_to_benchmark)
        num_proj_op_types = len(self.convex_proj_ops_to_benchmark)
        colormap = plt.cm.rainbow
        colors = [colormap(i) for i in np.linspace(0, 1, num_solvers)]

        # Plot the time for all cases vs the matrix size.
        fig, axs = plt.subplots(1, num_proj_op_types, sharey='row',
                                gridspec_kw={'hspace': 0, 'wspace': 0})

        for proj_type_id, proj_ops in enumerate(self.convex_proj_ops_to_benchmark):
            for solver_id, solver in enumerate(self.solvers_to_benchmark):
                # post process time
                num_problems = len(self.problem_sizes)
                mean = np.zeros(num_problems)
                lower_95 = np.zeros(num_problems)
                upper_95 = np.zeros(num_problems)
                for i in range(num_problems):
                    mean[i] = np.mean(
                        date[solver_id, proj_type_id, i, :])
                    sem = st.sem(date[solver_id, proj_type_id, i, :])
                    h = sem * st.t.ppf((1 + 0.95) / 2., num_problems - 1)
                    lower_95[i] = mean[i] - h
                    upper_95[i] = mean[i] + h

                axs[proj_type_id].plot(
                    self.problem_sizes, mean, label=solver.name, color=colors[solver_id])
                # axs[proj_type_id].fill_between(
                #     self.problem_sizes, lower_95, upper_95, alpha=0.2, color=colors[solver_id])
                # axs[proj_type_id].set_yscale('log')
                axs[proj_type_id].label_outer()

            axs[proj_type_id].set_title(
                proj_ops[0].name)

            if proj_type_id == 0:
                axs[proj_type_id].set_ylabel(name)

        plt.legend()
        plt.show()

    def process_results(self):
        self.plot('wall-clock time [s]', self._problem_time)
        self.plot('number of matrix-vector multiplications', self._problem_num_matrix_vector_mults)
        self.plot('solution residual', self._problem_residual)

        # num_solvers = len(self.solvers_to_benchmark)
        # num_proj_op_types = len(self.convex_proj_ops_to_benchmark)
        # colormap = plt.cm.rainbow
        # colors = [colormap(i) for i in np.linspace(0, 1, num_solvers)]

        # # Plot the time for all cases vs the matrix size.
        # fig, axs = plt.subplots(1, num_proj_op_types, sharey='row',
        #                         gridspec_kw={'hspace': 0, 'wspace': 0})

        # for proj_type_id, proj_ops in enumerate(self.convex_proj_ops_to_benchmark):
        #     for solver_id, solver in enumerate(self.solvers_to_benchmark):
        #         # post process time
        #         num_problems = len(self.problem_sizes)
        #         mean = np.zeros(num_problems)
        #         lower_95 = np.zeros(num_problems)
        #         upper_95 = np.zeros(num_problems)
        #         for i in range(num_problems):
        #             mean[i] = np.mean(
        #                 self._problem_time[solver_id, proj_type_id, i, :])
        #             sem = st.sem(self._problem_time[solver_id, proj_type_id, i, :])
        #             h = sem * st.t.ppf((1 + 0.95) / 2., num_problems - 1)
        #             lower_95[i] = mean[i] - h
        #             upper_95[i] = mean[i] + h

        #         axs[proj_type_id].plot(
        #             self.problem_sizes, mean, label=solver.name, color=colors[solver_id])
        #         # axs[proj_type_id].fill_between(
        #         #     self.problem_sizes, lower_95, upper_95, alpha=0.2, color=colors[solver_id])
        #         # axs[proj_type_id].set_yscale('log')
        #         axs[proj_type_id].label_outer()

        #     axs[proj_type_id].set_title(
        #         proj_ops[0].name)

        #     if proj_type_id == 0:
        #         axs[proj_type_id].set_ylabel(
        #             'wall-clock time [s]')

        # plt.legend()
        # plt.show()


        # # Plot the num mv mults for all cases vs the matrix size.
        # fig, axs = plt.subplots(1, num_proj_op_types, sharey='row',
        #                         gridspec_kw={'hspace': 0, 'wspace': 0})

        # for proj_type_id, proj_ops in enumerate(self.convex_proj_ops_to_benchmark):
        #     for solver_id, solver in enumerate(self.solvers_to_benchmark):
        #         # post process time
        #         num_problems = len(self.problem_sizes)
        #         mean = np.zeros(num_problems)
        #         lower_95 = np.zeros(num_problems)
        #         upper_95 = np.zeros(num_problems)
        #         for i in range(num_problems):
        #             mean[i] = np.mean(
        #                 self._problem_num_matrix_vector_mults[solver_id, proj_type_id, i, :])
        #             conf = st.t.interval(alpha=0.95, df=num_problems-1,
        #                                  loc=mean[i], scale=st.sem(self._problem_num_matrix_vector_mults[solver_id, proj_type_id, i, :]))
        #             lower_95[i] = conf[0]
        #             upper_95[i] = conf[1]

        #         axs[proj_type_id].plot(
        #             self.problem_sizes, mean, label=solver.name, color=colors[solver_id])
        #         # axs[proj_type_id].fill_between(
        #         #     self.problem_sizes, lower_95, upper_95, alpha=0.2, color=colors[solver_id])
        #         # axs[proj_type_id].set_yscale('log')
        #         axs[proj_type_id].label_outer()

        #     axs[proj_type_id].set_title(
        #         proj_ops[0].name)

        #     if proj_type_id == 0:
        #         axs[proj_type_id].set_ylabel(
        #             'number of matrix-vector multiplications')

        # plt.legend()
        # plt.show()


        # # Plot the residual for all cases vs the matrix size.
        # fig, axs = plt.subplots(1, num_proj_op_types, sharey='row',
        #                         gridspec_kw={'hspace': 0, 'wspace': 0})

        # for proj_type_id, proj_ops in enumerate(self.convex_proj_ops_to_benchmark):
        #     for solver_id, solver in enumerate(self.solvers_to_benchmark):
        #         # post process time
        #         num_problems = len(self.problem_sizes)
        #         mean = np.zeros(num_problems)
        #         lower_95 = np.zeros(num_problems)
        #         upper_95 = np.zeros(num_problems)
        #         for i in range(num_problems):
        #             mean[i] = np.mean(
        #                 self._problem_residual[solver_id, proj_type_id, i, :])
        #             conf = st.t.interval(alpha=0.95, df=num_problems-1,
        #                                  loc=mean[i], scale=st.sem(self._problem_residual[solver_id, proj_type_id, i, :]))
        #             lower_95[i] = conf[0]
        #             upper_95[i] = conf[1]

        #         axs[proj_type_id].plot(
        #             self.problem_sizes, mean, label=solver.name, color=colors[solver_id])
        #         # axs[proj_type_id].fill_between(
        #         #     self.problem_sizes, lower_95, upper_95, alpha=0.2, color=colors[solver_id])
        #         # axs[proj_type_id].set_yscale('log')
        #         axs[proj_type_id].label_outer()

        #     axs[proj_type_id].set_title(
        #         proj_ops[0].name)

        #     if proj_type_id == 0:
        #         axs[proj_type_id].set_ylabel(
        #             'solution residual')

        # plt.legend()
        # plt.show()

        print("")


if __name__ == '__main__':
    problem_sizes = np.linspace(2, 500, 50, dtype=int)
    num_random_trials = 10
    desired_tol = 1e-5
    max_mv_mults = 5000

    solvers_to_benchmark = [
        solvers.CCQPSolverAPGD(desired_tol, max_mv_mults),
        solvers.CCQPSolverAPGDAntiRelaxation(desired_tol, max_mv_mults),
        solvers.CCQPSolverBBPGD(desired_tol, max_mv_mults),
        solvers.CCQPSolverBBPGDf(desired_tol, max_mv_mults),
        solvers.CCQPSolverSPG(desired_tol, max_mv_mults)]

    proj_ops_to_benchmark = []
    proj_ops_to_benchmark.append([ss.IdentityProjOp(dim)
                                 for dim in problem_sizes])
    proj_ops_to_benchmark.append([ss.LowerBoundProjOp(dim)
                                  for dim in problem_sizes])
    proj_ops_to_benchmark.append([ss.UpperBoundProjOp(dim)
                                  for dim in problem_sizes])
    proj_ops_to_benchmark.append([ss.BoxProjOp(dim) for dim in problem_sizes])
    proj_ops_to_benchmark.append([ss.SphereProjOp(dim)
                                 for dim in problem_sizes])

    benchmark = BenchmarkRandomCCQP(
        num_random_trials, solvers_to_benchmark, proj_ops_to_benchmark)
    benchmark.run()
    benchmark.process_results()
