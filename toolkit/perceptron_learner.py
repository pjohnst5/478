from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix


class PerceptronLearner(SupervisedLearner):
    """
    This is the Perceptron Learner
    """

    labels = []

    def __init__(self):
        pass

    def train(self, features, labels):
        #add a bias column to features, make it 1

        #make a weights matrix, add one extra row for bias node

        # go through inputs line by line, multiplying by weights

        # check if it was right, if so do nothing

        # if not right, change the weights

        pass

    def predict(self, features, labels):
        pass
