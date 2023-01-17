import unittest
import os.path as osp

from tsp.logger import Logger


if __name__ == "__main__":
    test_dir = osp.dirname(__file__)
    testsuite = unittest.TestLoader().discover(test_dir)

    Logger.dummy_init()
    unittest.TextTestRunner(verbosity=2).run(testsuite)
