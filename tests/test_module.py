import unittest

from ccqppy.solution_spaces import IdentityProjOp

class TestSimple(unittest.TestCase):
    def test_example(self):
        convex_proj_op = IdentityProjOp()
        self.assertEqual(convex_proj_op(5) + convex_proj_op(6), 11)

if __name__ == '__main__':
    unittest.main()
