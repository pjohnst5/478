from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix

import numpy as np
import math


class PerceptronLearner(SupervisedLearner):
    """
    This is the Perceptron Learner
    """

    def __init__(self):
        pass

    def train(self, features, labels):
        self.debug = False
        # add a bias column to features, make it 1
        inputs = np.ones((features.rows, features.cols + 1), dtype=float)
        inputs[:,:-1] = features.data

        #see how many output classes there are
        self.classCount = labels.value_count(0)

        # make the weightSets, add one extra column to each weightSet for bias weight
        if self.classCount == 2:
            self.weights = np.random.uniform(low=-0.0, high=0.0, size=(1, features.cols+1))
        else:
            self.weights = np.random.uniform(low=-0.0, high=0.0, size=(self.classCount, features.cols+1))

        if self.debug:
            print("inputs\n", inputs, "\n")
            print("labels\n", labels.data, "\n")
            print("classCount\n", self.classCount, "\n")
            print("weights\n", self.weights, "\n")

        row = 0
        numCorrect = 0
        numTests = len(inputs) * len(self.weights)
        oldAccuracy = 0
        newAccuracy = 0
        numEpochs = 0
        done = False
        numEpochsNoImprovement = 0
        learningRate = 0.1
        accuracyThreshold = 0.01 #Any difference > than this is significant improvement
        epochThreshold = 5 #After this many epochs with no improvement we stop

        SSE = 0.0
        errors = []
        epochs = []

        while not done:
            if self.debug:
                print("----------------------------------------------\n")
                print("                   EPOCH ", numEpochs, "      \n")
                print("----------------------------------------------\n")
            #get the input row
            singleInput = inputs[row]

            #go through each "perceptron" and train on the input
            for weightSetIndex in range(len(self.weights)):
                target = 0
                if self.classCount == 2:
                    target = labels.row(row)[0]
                else:
                    if weightSetIndex == labels.row(row)[0]:
                        target = 1
                    else:
                        target = 0

                #get the net
                net = np.dot(singleInput, self.weights[weightSetIndex])

                #create output
                output = 0
                if net > 0:
                    output = 1
                else:
                    output = 0

                if self.debug:
                    print("-------\n")
                    print("weight set: ", weightSetIndex, "\n")
                    print('single input\n', singleInput, "\n")
                    print("label: ", labels.row(row)[0], "\n")
                    print('target\n', target, "\n")
                    print('self.weights\n', self.weights, "\n")
                    print('net\n', net, "\n")
                    print('output\n', output, "\n")

                # check if it was right, if so do nothing
                if output == target:
                    numCorrect += 1
                    if self.debug:
                        print('Matched\n')

                # if not right, change the weights
                else:
                    SSE += abs(target - output)**2
                    delta = learningRate * (target - output)
                    changeInweightSet = delta * singleInput

                    #adjust the weights
                    self.weights[weightSetIndex] = self.weights[weightSetIndex] + changeInweightSet

                    if self.debug:
                        print('No match\n')
                        print('delta\n', delta, "\n")
                        print('change in weightSet\n', changeInweightSet, "\n")
                        print('new self.weights\n', self.weights, "\n")

            #go to next row of inputs
            row += 1

            #if at the end of an epoch
            if row == len(inputs):
                numEpochs += 1
                #compute new accuracy
                newAccuracy = float(numCorrect / numTests)

                #find eror, map to numEpochs - 1
                MSE = SSE / numTests
                RMSE = math.sqrt(MSE)
                errors.append(RMSE)
                epochs.append(numEpochs-1)
                SSE = 0

                if self.debug:
                    print('END OF EPOCH: ', numEpochs, "\n")
                    print("numcorrect\n", numCorrect, "\n")
                    print("numTests\n", numTests, "\n")
                    print("newAccuracy\n", newAccuracy, "\n")

                #If no significant change in accuracy
                if abs(oldAccuracy - newAccuracy) < accuracyThreshold:
                    numEpochsNoImprovement += 1

                #Significant change was made
                else:
                    numEpochsNoImprovement = 0

                if numEpochsNoImprovement == epochThreshold:
                    if self.debug:
                        print('stopping')
                    done = True

                #shuffle training deck
                features.shuffle(labels)
                inputs[:,:-1] = features.data

                #reset variables
                row = 0
                numCorrect = 0
                oldAccuracy = newAccuracy
                newAccuracy = 0

        if self.debug:
            print(self.weights)
            print(numEpochs)
            print("Errors and epochs\n")
            print(errors)
            print()
            print(epochs)
        print(self.weights)

        pass

    def predict(self, features, labels):
        del labels[:]

        #add a bias weight onto the input
        features.append(1.0)

        nets = []
        for weightSetIndex in range(len(self.weights)):
            #find the net
            nets.append(np.dot(features, self.weights[weightSetIndex]))

        #find greatest net
        greatestNetIndex = 0
        for i in range(len(nets)):
            if nets[i] > nets[greatestNetIndex]:
                greatestNetIndex = i

        #create output
        output = 0
        if self.classCount == 2:
            if nets[greatestNetIndex] > 0:
                output = 1
            else:
                output = 0
        else:
            if nets[greatestNetIndex] > 0:
                output = greatestNetIndex


        if len(labels) == 0:
            labels.append(output)
        else:
            labels[0] = output

        if self.debug:
            print("features: ", features, "\n")
            print("output: ", output, "\n")
            print("nets: ", nets, "\n")
            print("weights: ", self.weights, "\n")

        pass
