from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix

import numpy as np
import math


class NeuralNetLearner(SupervisedLearner):
    """
    This is the Neural Net Learner
    """

    def __init__(self):
        self.debug = True
        pass

    def createNetwork(self, out, hid, inp):
        #set number of nodes
        NUM_OUTPUT_NODES = out
        NUM_HIDDEN_NODES = hid #(including bias node)
        NUM_INPUT_NODES = inp
        NUM_TOTAL_NODES = NUM_OUTPUT_NODES + NUM_HIDDEN_NODES + NUM_INPUT_NODES

        #make arrays to store node indexes
        self.outputIndexes = []
        self.hiddenIndexes = []
        self.inputIndexes = []
        for i in range(NUM_OUTPUT_NODES):
            self.outputIndexes.append(i)
        for i in range(NUM_OUTPUT_NODES, NUM_OUTPUT_NODES + NUM_HIDDEN_NODES):
            self.hiddenIndexes.append(i)
        for i in range(NUM_OUTPUT_NODES + NUM_HIDDEN_NODES, NUM_TOTAL_NODES):
            self.inputIndexes.append(i)

        #make weight matrix
        self.weights = np.random.uniform(low=-0.0, high=0.0, size=(NUM_TOTAL_NODES, NUM_TOTAL_NODES))

        #set weights for example1
        self.weights[1,0] = -0.01
        self.weights[2,0] = 0.03
        self.weights[3,0] = 0.02
        self.weights[4,0] = 0.02
        self.weights[5,1] = -0.03
        self.weights[6,1] = 0.03
        self.weights[7,1] = -0.01
        self.weights[5,2] = 0.04
        self.weights[6,2] = -0.02
        self.weights[7,2] = 0.01
        self.weights[5,3] = 0.03
        self.weights[6,3] = 0.02
        self.weights[7,3] = -0.02

        #initialize output map
        self.output = {}
        #initialize error map
        self.error = {}

    def f_net(self, net):
        result = 1 / (1 + (math.exp(-net)))
        return result

    def f_prime(self, net):
        result = net * (1 - net)
        return result

    def forwardProp(self, input):
        print("\ninput\n", input, "\n")

        #make hidden bias node always output 1
        self.output[self.hiddenIndexes[-1]] = 1

        #Calculate hidden layer outputs
        for num in self.hiddenIndexes[:-1]:
            #get weights going into num
            weightsIntoNum = self.weights[self.inputIndexes, num]
            #get the net for hidden node
            net = np.dot(input, weightsIntoNum)
            self.output[num] = self.f_net(net)

        #Calculate output layer outputs
        for num in self.outputIndexes:
            #get weights going into num
            weightsIntoNum = self.weights[self.hiddenIndexes, num]
            #get output from hidden layer
            hiddenLayerOutputs = []
            for hiddenNum in self.hiddenIndexes:
                hiddenLayerOutputs.append(self.output[hiddenNum])
            #get the net for the output node
            net = np.dot(hiddenLayerOutputs, weightsIntoNum)
            self.output[num] = self.f_net(net)

        for key in self.output:
            print("out_", key, ": ", self.output[key])
        print("\n")

    def backProp(self, label):
        
        pass

    def train(self, features, labels):
        #Num output, hidden(including bias), input nodes(including bias)
        self.createNetwork(1, 4, features.cols+1)

        # add a bias column to features, make it 1
        inputs = np.ones((features.rows, features.cols + 1), dtype=float)
        inputs[:,:-1] = features.data

        done = False
        inputIndex = 0

        while not done:
            singleInput = inputs[inputIndex]
            label = labels.row(inputIndex)[0]

            self.forwardProp(singleInput)
            self.backProp()

            done = True

        pass

    def predict(self, features, labels):

        pass
