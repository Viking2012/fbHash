import unittest
from fbHash.core.fbHashS import fbhashS

class FbHashTest(unittest.TestCase):
    def test_fbhashS_returns_test(self):
        response = fbhashS()
        self.assertEqual(response,'fbhashS','fbash failed to return test')
