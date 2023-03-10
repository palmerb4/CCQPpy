import unittest

from ccqppy.module import identity_proj_op

class TestSimple(unittest.TestCase):
    def test_example(self):
        self.assertEqual(identity_proj_op(5) + identity_proj_op(6), 11)

if __name__ == '__main__':
    unittest.main()
