from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix

import numpy as np


class InstanceBasedLearner(SupervisedLearner):
    """
    This is the nearest neighbor Learner
    """
    def __init__(self):
        pass

    def printFeatures(self):
        print("\nFeatures: ",)
        for r in range(len(self.features.data)):
            string = "["
            for c in range(len(self.features.data[r])):
                string += str(self.features.data[r][c])
                string += ", "
            string += "]"
            print(string)
        print()

    #returns two maps
    #first is data index to distance
    #second is data index to output class
    def findNeighbors(self, k, input):
        for i in range(self.features.rows):
            #deep copies input
            inputc = input[:]
            #deep copies the data row
            datac = self.features.data[i][:]

            for j in range(self.features.cols):
                #if you know both attributes (neither is inf)
                if inputc[j] != float("inf") and datac[j] != float("inf"):
                    




        pass

    def kNearPred(self, k):

        pass

    def kNearWeightPred(self, k):
        pass

    def kNearRegPred(self, k):
        pass

    def kNearRegWeightPred(self, k):
        pass

    def train(self, features, labels):
        self.features = features
        self.labels = labels

    def predict(self, features, labels):
        REG = False
        WEIGHTED = False
        K = 3
        prediction = None

        self.findNeighbors(K, features)

        if not REG and not WEIGHTED:
            prediction = self.kNearPred(K)
        elif not REG and WEIGHTED:
            prediction = self.kNearWeightPred(K)
        elif REG and not WEIGHTED:
            prediction = self.kNearRegPred(K)
        elif REG and WEIGHTED:
            prediction = self.kNearRegWeightPred(K)

        del labels[:]
        prediction = 0
        labels.append(prediction)
