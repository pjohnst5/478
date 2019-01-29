from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix

import numpy as np


class PerceptronLearner(SupervisedLearner):
    """
    This is the Perceptron Learner
    """

    labels = []

    def __init__(self):
        pass

    def train(self, features, labels):
        # add a bias column to features, make it 1
        inputs = np.ones((features.rows, features.cols + 1), dtype=float)
        inputs[:,:-1] = features.data
        print("inputs\n", inputs, "\n")

        # make a weights matrix, add one extra row for bias weight
        weights = np.random.uniform(low=-0.0, high=0.0, size=(features.cols+1, 1))

        # go through features row by row, multiplying that row by weights matrix
        row = 0
        numCorrect = 0
        numTotal = len(inputs)
        oldAccuracy = 0
        newAccuracy = 0
        numEpochs = 0
        done = False
        learningRate = 0.1
        accuracyThreshold = 0.05 #stops learning if accuracy is less than threshold
        # stops when the change in accuracy is less than the desired threshold, checks every epoch
        while not done:
            #get the row
            singleInput = inputs[row]
            #get the target
            target = labels.data[row][0]
            print('single input\n', singleInput, "\n")
            print('target\n', target, "\n")
            print('weights\n', weights, "\n")
            net = np.dot(singleInput, weights)
            print('net\n', net, "\n")

            #create output
            output = 0
            if net > 0:
                output = 1
            else:
                output = 0
            print('output\n', output, "\n")

            # check if it was right, if so do nothing
            if output == target:
                numCorrect += 1
                print('Matched\n')

            # if not right, change the weights
            else:
                print('No match\n')
                delta = learningRate * (target - output)
                print('delta\n', delta, "\n")

                changeInWeights = delta * singleInput
                print('change in weights\n', changeInWeights, "\n")

                #got through weight by weight adjusting the weights
                weights = weights + changeInWeights
                print('new weights\n', weights, "\n")

            #go to next row of inputs
            row += 1

            #if at the end of an epoch
            if row == numTotal:
                numEpochs += 1
                print('END OF EPOCH: ', numEpochs, "\n")
                #compute new accuracy
                newAccuracy = float(numCorrect / numTotal)
                print("numcorrect\n", numCorrect, "\n")
                print("numTotal\n", numTotal, "\n")
                print("newAccuracy\n", newAccuracy, "\n")

                #if no significant change in accuracy, stop
                if abs(oldAccuracy - newAccuracy) < accuracyThreshold:
                    print('stopping')
                    done = True
                else:
                    #reset variables
                    row = 0
                    numCorrect = 0
                    oldAccuracy = newAccuracy
                    newAccuracy = 0

        pass

    def predict(self, features, labels):
        pass
