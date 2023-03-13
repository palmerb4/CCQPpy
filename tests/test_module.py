# external
import unittest
import numpy as np

# internal
from ccqppy import solution_spaces as ss

class TestSolutionSpaces(unittest.TestCase):
    def test_identity(self):
        num_unknowns = 10
        convex_proj_op = ss.IdentityProjOp(num_unknowns)
        x_random = np.random.rand(num_unknowns)
        self.assertTrue(np.all(convex_proj_op(x_random) == x_random))

if __name__ == '__main__':
    unittest.main()
