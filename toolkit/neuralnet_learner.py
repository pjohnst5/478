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
        #Learning rate and momentum
        self.learningRate = 0.175
        self.momentum = 0.9

        #set number of nodes
        NUM_OUTPUT_NODES = out
        NUM_HIDDEN_NODES = hid #(including bias node)
        NUM_INPUT_NODES = inp
        NUM_TOTAL_NODES = NUM_OUTPUT_NODES + NUM_HIDDEN_NODES + NUM_INPUT_NODES

        #make arrays to store node indexes and targets
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
        #make delta matrix
        self.delta = np.random.uniform(low=-0.0, high=0.0, size=(NUM_TOTAL_NODES, NUM_TOTAL_NODES))

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
        #make input nodes have output of the input
        tempIndex = 0
        for inp in self.inputIndexes[:-1]:
            self.output[inp] = input[tempIndex]
            tempIndex += 1
        self.output[self.inputIndexes[-1]] = 1

        #Calculate hidden layer outputs (not including bias)
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
        #Set up target array
        self.targets = np.zeros(len(self.outputIndexes))
        #If it's continuous, keep the label as is
        if len(self.targets) == 1:
            self.targets[0] = label
        #if it's nominal, set the node who is supposed to say yes to 1, all others 0
        else:
            label = int(label)
            self.targets[label] = 1
        print(self.targets)
        print(label)

        #calculate output layer error
        for num in self.outputIndexes:
            self.error[num] = (self.targets[num] - self.output[num]) * self.f_prime(self.output[num])
            #calculate change in weights going into this output node
            for hid in self.hiddenIndexes:
                self.delta[hid, num] = (self.learningRate * self.error[num] * self.output[hid]) + (self.momentum * self.delta[hid,num])

        #calculate hidden layer error (not including bias)
        for num in self.hiddenIndexes[:-1]:
            #sum up errors of layer in front timesed by weight (output layer)
            sum = 0
            for out in self.outputIndexes:
                next = self.error[out] * self.weights[num,out]
                sum = sum + next
            #times this sum by fprime
            self.error[num] = sum * self.f_prime(self.output[num])
            #calculate change in weights going into this hidden node
            for inp in self.inputIndexes:
                self.delta[inp, num] = (self.learningRate * self.error[num] * self.output[inp]) + (self.momentum * self.delta[inp,num])

        #update weights
        self.weights = self.weights + self.delta

        for key in self.error:
            print("e_", key, ": ", self.error[key])
        print("\n")

        print("w_0=", self.weights[4,0])
        print("w_1=", self.weights[1,0])
        print("w_2=", self.weights[2,0])
        print("w_3=", self.weights[3,0])
        print("w_4=", self.weights[7,1])
        print("w_5=", self.weights[5,1])
        print("w_6=", self.weights[6,1])
        print("w_7=", self.weights[7,2])
        print("w_8=", self.weights[5,2])
        print("w_9=", self.weights[6,2])
        print("w_10=", self.weights[7,3])
        print("w_11=", self.weights[5,3])
        print("w_12=", self.weights[6,3], "\n")

    def train(self, features, labels):
        outputNodeCount = 0
        #if the value count is 0, it's continuous, 1 output node
        if labels.value_count(0) == 0:
            outputNodeCount = 1
        #else there are multiple classes, make that many output nodes
        else:
            outputNodeCount = labels.value_count(0)

        #Num output, hidden(including bias), input nodes(including bias)
        self.createNetwork(outputNodeCount, 4, features.cols+1)

        # add a bias column to features, make it 1
        inputs = np.ones((features.rows, features.cols + 1), dtype=float)
        inputs[:,:-1] = features.data

        done = False
        inputIndex = 0
        hey = 0

        while not done:
            singleInput = inputs[inputIndex]
            label = labels.row(inputIndex)[0]

            self.forwardProp(singleInput)
            self.backProp(label)

            if inputIndex == 2:
                done = True
            inputIndex += 1

        pass

    def predict(self, features, labels):

        pass
