from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix


class InstanceBasedLearner(SupervisedLearner):
    """
    This is the nearest neighbor Learner
    """
    def __init__(self):
        pass

    def findNeighbors(k, input):
        pass

    def kNearPred(k):
        pass

    def kNearWeightPred(k):
        pass

    def kNearRegPred(k):
        pass

    def kNearRegWeightPred(k):
        pass

    def train(self, features, labels):
        print(features.data)
        pass

    def predict(self, features, labels):
        del labels[:]
        prediction = 0
        labels.append(prediction)
