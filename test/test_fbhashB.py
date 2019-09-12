import unittest
from fbHash.core.fbHashB import fbhashB

class FbHashTest(unittest.TestCase):
    def test_fbhash_returns_test(self):
        response = fbhashB()
        self.assertEqual(response,'test','fbash failed to return test')
        