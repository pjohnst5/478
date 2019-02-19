from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix

import numpy as np
import math
import copy


class NeuralNetLearner(SupervisedLearner):
    """
    This is the Neural Net Learner
    """

    def __init__(self):
        self.debug = True
        pass

    def createNetwork(self, nodes):
        #the first int in nodes is number of output nodes
        #the last int in nodes is the number of input nodes
        #everything in between is the number of nodes for that hidden layer
        #Learning rate and momentum
        self.learningRate = 0.175 #0.1
        self.momentum = 0.9 #0
        #lets go
        NUM_TOTAL_NODES = sum(nodes)
        #array of arrays to store node indexes
        self.nodeIndexes = []
        nodeID = 0
        for i in range(len(nodes)):
            numNodesForThisLayer = nodes[i]
            temp = []
            for j in range(numNodesForThisLayer):
                temp.append(nodeID)
                nodeID += 1
            self.nodeIndexes.append(copy.deepcopy(temp))
            temp.clear()

        #make weight matrix
        self.weights = np.random.uniform(low=-0.0, high=0.0, size=(NUM_TOTAL_NODES, NUM_TOTAL_NODES))
        #make delta matrix
        self.delta = np.random.uniform(low=-0.0, high=0.0, size=(NUM_TOTAL_NODES, NUM_TOTAL_NODES))

        #initialize output map
        self.output = {}
        #initialize error map
        self.error = {}
        #set all bias nodes output (for hidden layers) to 1
        hiddenLayers = self.nodeIndexes[1:-1]
        for i in range(len(hiddenLayers)):
            nodeID = hiddenLayers[i][-1]
            self.output[nodeID] = 1

        #set weights for example 2
        # self.weights[10,6] = 0.1
        # self.weights[9,6] = 0.2
        # self.weights[8,6] = -0.1
        # self.weights[10,5] = -0.2
        # self.weights[9,5] = 0.3
        # self.weights[8,5] = -0.3
        # self.weights[7,3] = 0.1
        # self.weights[6,3] = -0.2
        # self.weights[5,3] = -0.3
        # self.weights[7,2] = 0.2
        # self.weights[6,2] = -0.1
        # self.weights[5,2] = 0.3
        # self.weights[4,1] = 0.2
        # self.weights[3,1] = -0.1
        # self.weights[2,1] = 0.3
        # self.weights[4,0] = 0.1
        # self.weights[3,0] = -0.2
        # self.weights[2,0] = -0.3
        # print("Weights:")
        # print(self.weights[10,6], self.weights[9,6], self.weights[8,6])
        # print(self.weights[10,5], self.weights[9,5], self.weights[8,5])
        # print(self.weights[7,3], self.weights[6,3], self.weights[5,3])
        # print(self.weights[7,2], self.weights[6,2], self.weights[5,2])
        # print(self.weights[4,1], self.weights[3,1], self.weights[2,1])
        # print(self.weights[4,0], self.weights[3,0], self.weights[2,0])

        # #set weights for example1
        # self.weights[1,0] = -0.01
        # self.weights[2,0] = 0.03
        # self.weights[3,0] = 0.02
        # self.weights[4,0] = 0.02
        # self.weights[5,1] = -0.03
        # self.weights[6,1] = 0.03
        # self.weights[7,1] = -0.01
        # self.weights[5,2] = 0.04
        # self.weights[6,2] = -0.02
        # self.weights[7,2] = 0.01
        # self.weights[5,3] = 0.03
        # self.weights[6,3] = 0.02
        # self.weights[7,3] = -0.02


    def f_net(self, net):
        result = 1 / (1 + (math.exp(-net)))
        return result

    def f_prime(self, net):
        result = net * (1 - net)
        return result

    def forwardProp(self, input):
        print("\ninput\n", input, "\n")

        #make input nodes have output of the input
        tempIndex = 0
        for inp in self.nodeIndexes[-1][:-1]:
            self.output[inp] = input[tempIndex]
            tempIndex += 1
        self.output[self.nodeIndexes[-1][-1]] = 1

        #Calculate each layers output (starting with input layer)
        for layerIndex in reversed(range(len(self.nodeIndexes) - 1)):
            layerIds = []
            #if it's the output layer
            if layerIndex == 0:
                #get the node id's for this layer (whole layer because there's no bias node)
                layerIds = self.nodeIndexes[layerIndex]

            #if it's a hidden layer
            else:
                #get the node id's for this layer (not including bias)
                layerIds = self.nodeIndexes[layerIndex][:-1]

            #get the node id's of the layer before it (including bias)
            layerBeforeItIds = self.nodeIndexes[layerIndex+1]
            #get the output of the layer before it (including bias)
            outputOfBeforeLayer = []
            for i in layerBeforeItIds:
                outputOfBeforeLayer.append(self.output[i])
            #print(layerIndex, layerIds, layerBeforeItIds, outputOfBeforeLayer)
            #Calculate this layer's outputs
            for num in layerIds:
                #get weights going into num
                weightsIntoNum = self.weights[layerBeforeItIds, num]
                #get the net for hidden node
                net = np.dot(outputOfBeforeLayer, weightsIntoNum)
                self.output[num] = self.f_net(net)


        for key in self.output:
            print("out_", key, ": ", self.output[key])
        print("\n")


    def backProp(self, label):
        #Set up target array
        self.targets = np.zeros(len(self.nodeIndexes[0]))
        # self.targets[0] = 1
        # self.targets[1] = 0.1
        #If it's continuous, keep the label as is
        if len(self.targets) == 1:
            self.targets[0] = label
        #if it's nominal, set the node who is supposed to say yes to 1, all others 0
        else:
            self.targets[int(label)] = 1

        #calculae each layer's error (starting with output layer)
        for layerIndex in range(len(self.nodeIndexes) - 1):
            #if it's the output layer
            if layerIndex == 0:
                #get the node id's for this layer (whole layer because there's no bias node)
                layerIds = self.nodeIndexes[layerIndex]
                #get the node id's of the layer before it (including bias)
                layerBeforeItIds = self.nodeIndexes[layerIndex+1]
                #calculate output layer error
                for num in layerIds:
                    self.error[num] = (self.targets[num] - self.output[num]) * self.f_prime(self.output[num])
                    #calculate change in weights going into this output node
                    for bef in layerBeforeItIds:
                        self.delta[bef, num] = (self.learningRate * self.error[num] * self.output[bef]) + (self.momentum * self.delta[bef,num])

            #if it's a hidden layer
            else:
                #get the node id's for this layer (not including bias)
                layerIds = self.nodeIndexes[layerIndex][:-1]
                #get the node id's of the layer before it (including bias)
                layerBeforeItIds = self.nodeIndexes[layerIndex+1]
                #get the node id's of the layer IN FRONT of it
                layerAfterItIds = []
                if layerIndex == 1:
                    layerAfterItIds = self.nodeIndexes[layerIndex-1]
                else:
                    layerAfterItIds = self.nodeIndexes[layerIndex-1][:-1]
                #caculate hidden layer error (not including bias)
                for num in layerIds:
                    #sum up errors of layer in front timessed by weight
                    sum = 0
                    for aft in layerAfterItIds:
                        next = self.error[aft] * self.weights[num, aft]
                        sum = sum + next
                    #times this sum by fprime
                    self.error[num] = sum * self.f_prime(self.output[num])
                    #calculate change in weights going into this hidden node
                    for bef in layerBeforeItIds:
                        self.delta[bef,num] = (self.learningRate * self.error[num] * self.output[bef]) + (self.momentum * self.delta[bef,num])
        #update weights
        self.weights = self.weights + self.delta

        for key in self.error:
            print("e_", key, ": ", self.error[key])
        print("\n")
        #print weights for example 2
        # print("Weights:")
        # print(self.weights[10,6], self.weights[9,6], self.weights[8,6])
        # print(self.weights[10,5], self.weights[9,5], self.weights[8,5])
        # print(self.weights[7,3], self.weights[6,3], self.weights[5,3])
        # print(self.weights[7,2], self.weights[6,2], self.weights[5,2])
        # print(self.weights[4,1], self.weights[3,1], self.weights[2,1])
        # print(self.weights[4,0], self.weights[3,0], self.weights[2,0])

        #print weights for example 1
        # print("w_0=", self.weights[4,0])
        # print("w_1=", self.weights[1,0])
        # print("w_2=", self.weights[2,0])
        # print("w_3=", self.weights[3,0])
        # print("w_4=", self.weights[7,1])
        # print("w_5=", self.weights[5,1])
        # print("w_6=", self.weights[6,1])
        # print("w_7=", self.weights[7,2])
        # print("w_8=", self.weights[5,2])
        # print("w_9=", self.weights[6,2])
        # print("w_10=", self.weights[7,3])
        # print("w_11=", self.weights[5,3])
        # print("w_12=", self.weights[6,3], "\n")


    def train(self, features, labels):
        outputNodeCount = 0
        #if the value count is 0, it's continuous, 1 output node
        if labels.value_count(0) == 0:
            outputNodeCount = 1
        #else there are multiple classes, make that many output nodes
        else:
            outputNodeCount = labels.value_count(0)
        inputNodeCount = features.cols+1

        #First num in array is num of output, last is num of input
        #between are number of nodes for hidden layers (including bias)
        self.createNetwork([2, 3, 3, inputNodeCount])

        # add a bias column to features, make it 1
        inputs = np.ones((features.rows, features.cols + 1), dtype=float)
        inputs[:,:-1] = features.data

        done = False
        inputIndex = 0

        while not done:
            singleInput = inputs[inputIndex]
            label = labels.row(inputIndex)[0]

            self.forwardProp(singleInput)
            self.backProp(label)

            if inputIndex == 1:
                done = True
            inputIndex += 1

        pass

    def predict(self, features, labels):

        pass
