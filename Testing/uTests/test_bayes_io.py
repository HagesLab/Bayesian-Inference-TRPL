import unittest
import numpy as np
import os
import logging

from bayes_io import get_initpoints
eps0 = 8.854 * 1e-12 * 1e-9 # [C / V m] to {C / V nm}
q = 1.0 # [e]
q_C = 1.602e-19 # [C]
kB = 8.61773e-5  # [eV / K]

class TestUtils(unittest.TestCase):
    
    def setUp(self):
        self.logger = logging.getLogger()
        pass
    
    def test_get_initpoints(self):
        # Basic read
        ic_flags = {'select_obs_sets':None}
        where_inits = os.path.join("Testing", "uTests", "testfiles", "test_init_cond.txt")
        ic = get_initpoints(where_inits, ic_flags, scale_f=1)
       
        expected = np.array([[1,2,3,4,5], [5.1,4.1,3.1,2.1,1.1], [1e12, 1e13, 1e14, 1e13, 1e12]], dtype=float)
        np.testing.assert_equal(expected, ic)

        # Select one IC
        ic_flags = {'select_obs_sets':[1]}
        ic = get_initpoints(where_inits, ic_flags, scale_f=1)
       
        expected = np.array([[5.1,4.1,3.1,2.1,1.1]], dtype=float)
        np.testing.assert_equal(expected, ic)
        
        # Scale on way in
        sf = 42
        ic_flags = {'select_obs_sets':[1]}
        ic = get_initpoints(where_inits, ic_flags, scale_f=sf)
       
        expected = sf*np.array([[5.1,4.1,3.1,2.1,1.1]], dtype=float)
        np.testing.assert_equal(expected, ic)
        
        # Log message working?
        with self.assertLogs() as captured:
            get_initpoints(where_inits, ic_flags, scale_f=1, logger=self.logger)
        self.assertEqual(len(captured.records), 1) # check that there is only one log message
        self.assertEqual(captured.records[0].getMessage(), "Failed to read IE") # and it is the proper one
        
        return