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

        # self.printMatrix(self.features.data, "Features")
        # print("input")
        # print(input)
        # self.printMatrix(difference, "difference")

        #if difference is inf, -inf, or nan there was a unknown
        difference = np.where(difference == float("inf"), 1, difference)
        difference = np.where(difference == -float("inf"), 1, difference)
        difference = np.where(np.isnan(difference), 1, difference)

        # self.printMatrix(difference, "difference after taking out unknowns")

        #fix all nominal columns
        for j in range(self.features.cols):
            #if this attribute is nominal
            if self.features.value_count(j) != 0:
                difference[:, j] = np.where(difference[:, j] != 0, 1, 0)

        # self.printMatrix(difference, "difference after fixing nominal")

        #find the euclidean distance for each difference array
        distances = np.linalg.norm(difference, axis=-1)
        indices = np.argpartition(distances, k)[:k]

        indexToDist = {}
        indexToLabel = {}
        for i in indices:
            if distances[i] == 0:
                indexToDist[i] = 1e-10
            else:
                indexToDist[i] = distances[i]
            indexToLabel[i] = self.labels.data[i][0]

        # print("distances")
        # print(distances)
        # print("indices")
        # print(indices)
        # print("index to dist")
        # print(indexToDist)
        # print("index to label")
        # print(indexToLabel)
        return indexToDist, indexToLabel

    def kNearPred(self, k, input):
        _, indexToLabel = self.findNeighbors(k, input)
        counts = np.zeros((self.labels.value_count(0), ), dtype=int)

        for key in indexToLabel:
            counts[int(indexToLabel[key])] += 1

        prediction = np.argmax(counts)
        return prediction

    def kNearWeightPred(self, k, input):
        indexToDist, indexToLabel = self.findNeighbors(k, input)
        counts = np.zeros((self.labels.value_count(0), ), dtype=float)

        for key in indexToDist:
            invDistSq = 1 / (indexToDist[key]**2)
            try:
                counts[int(indexToLabel[key])] += invDistSq
            except OverflowError:
                counts[int(indexToLabel[key])] = float("inf")
        prediction = np.argmax(counts)
        return prediction

    def kNearRegPred(self, k, input):
        _, indexToLabel = self.findNeighbors(k, input)
        mean = 0
        for key in indexToLabel:
            mean += indexToLabel[key]
        mean = mean / len(indexToLabel)
        return mean

    def kNearRegWeightPred(self, k, input):
        indexToDist, indexToLabel = self.findNeighbors(k, input)
        top = 0
        bottom = 0

        for key in indexToDist:
            top += (indexToLabel[key] / (indexToDist[key]**2))
            bottom += (1 / (indexToDist[key]**2))

        prediction = top/bottom
        return prediction

    def train(self, features, labels):
        self.features = features
        self.labels = labels

    def predict(self, features, labels):
        REG = False
        WEIGHTED = True
        K = 15
        prediction = None

        if not REG and not WEIGHTED:
            prediction = self.kNearPred(K, features)
        elif not REG and WEIGHTED:
            prediction = self.kNearWeightPred(K, features)
        elif REG and not WEIGHTED:
            prediction = self.kNearRegPred(K, features)
        elif REG and WEIGHTED:
            prediction = self.kNearRegWeightPred(K, features)

        del labels[:]
        labels.append(prediction)
