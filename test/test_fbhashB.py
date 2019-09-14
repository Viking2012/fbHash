import unittest
from fbHash.core.fbHashB import fbhashB

class FbHashTest(unittest.TestCase):
    def test_fbhash_returns_test(self):
        response = fbhashB()
        self.assertEqual(response,[b'this is', b'his is ', b'is is a', b's is a ', b' is a t', b'is a te', b's a tes', b' a test'],'fbash failed to return test')
