from __future__ import (absolute_import, division, print_function, unicode_literals)

from .matrix import Matrix
import math

# this is an abstract class
class GroupLearner:

    def train(self, features, labels):
        """
        Before you call this method, you need to divide your data
        into a feature matrix and a label matrix.
        :type features: Matrix
        :type labels: Matrix
        """
        raise NotImplementedError()

    def predict(self, features, labels):
        """
        A feature vector goes in. A label vector comes out. (Some supervised
        learning algorithms only support one-dimensional label vectors. Some
        support multi-dimensional label vectors.)
        :type features: [float]
        :type labels: [float]
        """
        raise NotImplementedError

    def measure_accuracy(self, features, labels, confustion=None):

        if features.rows != labels.rows:
            raise Exception("Expected the features and labels to have the same number of rows")
        if labels.cols != 1:
            raise Exception("Sorry, this method currently only supports one-dimensional labels")
        if features.rows == 0:
            raise Exception("Expected at least one row")

        correctCount = 0

        #for every pair of instances in idea1test.arff
        for i in range(0, features.rows, 2):
            #first we need to see which of the two instances has 1 as the label
            team0Label = labels.data[i][0]
            team1Label = labels.data[i+1][0]
            print(team1Label, team0Label, "labels")
            print("features: ")
            print(features.row(i))
            print(features.row(i+1))


            #find which won this world series
            winningTeam = None
            if team0Label == 1:
                winningTeam = 0
            elif team1Label == 1:
                winningTeam = 1
            else:
                raise Exception("Neither team has 1 as their label in measure accuracy")

            #now predict on these two teams
            feat0 = features.row(i)
            feat1 = features.row(i+1)
            predictionTeam0 = []
            predictionTeam1 = []
            self.predict(feat0, predictionTeam0)
            self.predict(feat1, predictionTeam1)

            #find which prediction is higher
            predictedWinningTeam = None
            if predictionTeam0[0] >= predictionTeam1[0]:
                predictedWinningTeam = 0
            elif predictionTeam0[0] < predictionTeam1[0]:
                predictedWinningTeam = 1

            #see if we predicted correctly
            if winningTeam == predictedWinningTeam:
                correctCount += 1

        return correctCount / features.rows
