# external
import unittest
import numpy as np

# internal
from ccqppy import solution_spaces as ss

class TestSolutionSpaces(unittest.TestCase):
    def test_identity(self):
        convex_proj_op = ss.IdentityProjOp()
        x_random = np.random.rand(10)
        self.assertTrue(np.all(convex_proj_op(x_random) == x_random))

if __name__ == '__main__':
    unittest.main()
