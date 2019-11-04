import unittest
from ..src import spectrumUtilities as su
import numpy as np

class spectrumUtilityTests(unittest.TestCase):

    def test_glueEmpty(self):
        a = np.array([])
        b = np.array([])
        self.assertTrue(su.glueSpectra(a, b).size == 0)

    def test_glueOverlaping(self):
        a = np.array([1,2,3])
        b = np.array([2,3,4])
        c = su.glueSpectra(a, b)
        d = np.array([1,2,3,4])
        self.assertTrue(c.size == 4)
        self.assertTrue(all([ x == y for (x, y) in zip(c, d) ]))

    def test_glueNonoverlaping(self):
        a = np.array([1,2])
        b = np.array([3,4])
        c = su.glueSpectra(a, b)
        d = np.array([1,2,3,4])
        self.assertTrue(c.size == 4)
        self.assertTrue(all([ x == y for (x, y) in zip(c, d) ]))

    def test_glueContainingRight(self):
        a = np.array([1,2,3,4])
        b = np.array([3,4])
        c = su.glueSpectra(a, b)
        d = np.array([1,2,3,4])
        self.assertTrue(c.size == 4)
        self.assertTrue(all([ x == y for (x, y) in zip(c, d) ]))

    def test_glueContainingLeft(self):
        a = np.array([3,4])
        b = np.array([1,2,3,4])
        c = su.glueSpectra(a, b)
        d = np.array([1,2,3,4])
        self.assertTrue(c.size == 4)
        self.assertTrue(all([ x == y for (x, y) in zip(c, d) ]))

if __name__ == '__main__':
    unittest.main()