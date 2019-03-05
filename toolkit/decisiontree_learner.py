from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix

import numpy as np

class DecisionTreeLearner(SupervisedLearner):
    """
    This is the Neural Net Learner
    """

    def __init__(self):
        self.debug = True
        pass


    def train(self, features, labels):
        print("Hello")
        pass


    def predict(self, features, labels):
        pass
