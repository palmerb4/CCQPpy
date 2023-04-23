# external
import unittest
import numpy as np

# internal
from ccqppy import solution_spaces as ss
from ccqppy import solvers
from ccqppy import problem_suite

class TestSolutionSpaces(unittest.TestCase):
    def test_identity(self):
        num_unknowns = 10
        convex_proj_op = ss.IdentityProjOp(num_unknowns)
        x_random = np.random.rand(num_unknowns)
        self.assertTrue(np.all(convex_proj_op(x_random) == x_random))


class TestSolversAgainstSimpleProblems(unittest.TestCase):
    def test_APGD(self):
        test_problems = [problem_suite.UnconstrainedSPD1(),
                         problem_suite.UnconstrainedSPD2(),
                         problem_suite.BoxConstrainedSPD(),
                         problem_suite.ThinBoxConstrainedSPD(),
                         problem_suite.ActiveBoxConstrainedSPD()]
        
        for test_problem in test_problems:
            # Test APGD
            result = solvers.CCQPSolverAPGD(1e-8, 10000).solve(
                test_problem.A, test_problem.b, convex_proj_op=test_problem.convex_proj_op)
            self.assertTrue(result.solution_converged)
            self.assertTrue(np.linalg.norm(result.solution - test_problem.exact_solution) < 1e-5)

            # Test APGD with antirelaxation
            result = solvers.CCQPSolverAPGDAntiRelaxation(1e-8, 10000).solve(
                test_problem.A, test_problem.b, convex_proj_op=test_problem.convex_proj_op)
            self.assertTrue(result.solution_converged)
            self.assertTrue(np.linalg.norm(result.solution - test_problem.exact_solution) < 1e-5)

            # Test BBPGD
            result = solvers.CCQPSolverBBPGD(1e-8, 10000).solve(
                test_problem.A, test_problem.b, convex_proj_op=test_problem.convex_proj_op)
            self.assertTrue(result.solution_converged)
            self.assertTrue(np.linalg.norm(result.solution - test_problem.exact_solution) < 1e-5)

            # Test BBPGD with fallback
            result = solvers.CCQPSolverBBPGDf(1e-8, 10000).solve(
                test_problem.A, test_problem.b, convex_proj_op=test_problem.convex_proj_op)
            self.assertTrue(result.solution_converged)
            self.assertTrue(np.linalg.norm(result.solution - test_problem.exact_solution) < 1e-5)

            # Test SPG for PG
            result = solvers.CCQPSolverSPG(1e-8, 10000).solve(
                test_problem.A, test_problem.b, convex_proj_op=test_problem.convex_proj_op)
            self.assertTrue(result.solution_converged)
            self.assertTrue(np.linalg.norm(result.solution - test_problem.exact_solution) < 1e-5)
            

if __name__ == '__main__':
    # unittest.main()
    test = TestSolversAgainstSimpleProblems()
    test.test_APGD()