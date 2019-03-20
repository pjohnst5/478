from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix

import numpy as np
np.seterr(invalid='ignore')


class InstanceBasedLearner(SupervisedLearner):
    """
    This is the nearest neighbor Learner
    """
    def __init__(self):
        pass

    def printMatrix(self, matrix, title):
        print("\n",title)
        for r in range(len(matrix)):
            string = "["
            for c in range(len(matrix[r])):
                string += str(matrix[r][c])
                string += ", "
            string += "]"
            print(string)
        print()

    #returns two maps
    #first is data index to distance
    #second is data index to output class
    def findNeighbors(self, k, input):
        inputNP = np.array(input)
        difference = inputNP - self.features.data
        self.printMatrix(self.features.data, "Features")
        print("input")
        print(input)

        self.printMatrix(difference, "difference")

        #if hey is inf, -inf, or nan there was a unknown
        difference = np.where(difference == float("inf"), 1, difference)
        difference = np.where(difference == -float("inf"), 1, difference)
        difference = np.where(np.isnan(difference), 1, difference)
        self.printMatrix(difference, "difference after taking out unknowns")

        #fix all nominal columns
        for j in range(self.features.cols):
            #if this attribute is nominal
            if self.features.value_count(j) != 0:
                difference[:, j] = np.where(difference[:, j] != 0, 1, 0)

        self.printMatrix(difference, "difference after fixing nominal")

        #find the euclidean distance for each difference array
        distances = np.linalg.norm(difference, axis=-1)
        print("distances")
        print(distances)



        # for i in range(self.features.rows):
        #     #deep copies input
        #     inputc = input[:]
        #     #deep copies the data row
        #     datac = self.features.data[i][:]
        #
        #     for j in range(self.features.cols):
        #         #if you know both attributes (neither is inf)
        #         if inputc[j] != float("inf") and datac[j] != float("inf"):
        #             #if this attriute is nominal
        #             if self.features.value_count(j) != 0:
        #                 #if both attributes are the same
        #                 if datac[j] == inputc[j]:
        #                     inputc[j] = 0
        #                     datac[j] = 0
        #                 else:
        #                     inputc[j] = 0
        #                     datac[j] = 1
        #         #you don't know one or both
        #         else:
        #             inputc[j] = 0
        #             datac[j] = 1
        #
        #     #now find euclidean distance between inputc and datac







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
