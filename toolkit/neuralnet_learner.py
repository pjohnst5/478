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

    def createSets(self, features, labels, percent, vowel):
        #features.shuffle(labels)
        rowCountTrain = int(features.rows * (1-percent))
        print("features")
        features.print()
        labels.print()
        print(rowCountTrain, "\n")
        if vowel:
            self.trainFeatures = Matrix(features, 0, 3, rowCountTrain, features.cols - 3)
            self.validFeatures = Matrix(features, rowCountTrain, 3, features.rows-rowCountTrain, features.cols - 3)
        else:
            self.trainFeatures = Matrix(features, 0, 0, rowCountTrain, features.cols)
            self.validFeatures = Matrix(features, rowCountTrain, 0, features.rows-rowCountTrain, features.cols)
        self.trainLabels = Matrix(labels, 0, 0, rowCountTrain, labels.cols)
        self.validLables = Matrix(labels, rowCountTrain, 0, features.rows-rowCountTrain, features.cols)

        print("\nTrain features")
        self.trainFeatures.print()
        self.trainLabels
        print("\nValidation features")
        self.validFeatures.print()
        self.validLables.print()

        # add a bias column to features, make it 1
        self.trainInputs = np.ones((self.trainFeatures.rows, self.trainFeatures.cols + 1), dtype=float)
        self.trainInputs[:,:-1] = self.trainFeatures.data
        print()
        print(self.trainInputs)



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
        self.weights = np.random.uniform(low=-1.0, high=1.0, size=(NUM_TOTAL_NODES, NUM_TOTAL_NODES))
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


    def train(self, features, labels):
        #seperate out training, testing and validation sets
        #third argument is percent for validation, last is vowel
        self.createSets(features, labels, 0.20, True)

        inputNodeCount = features.cols+1
        outputNodeCount = 0
        #if the value count is 0, it's continuous, 1 output node
        if labels.value_count(0) == 0:
            outputNodeCount = 1
        #else there are multiple classes, make that many output nodes
        else:
            outputNodeCount = labels.value_count(0)

        #First num in array is num of output nodes, last is num of input nodes
        #numbers between are node counts for hidden layers (including bias)
        self.createNetwork([outputNodeCount, 3, 3, inputNodeCount])

        done = False
        inputIndex = 0

        while not done:
            singleInput = self.trainInputs[inputIndex]
            label = self.trainLabels.row(inputIndex)[0]

            self.forwardProp(singleInput)
            self.backProp(label)

            if inputIndex == 1:
                done = True
            inputIndex += 1

        pass

    def predict(self, features, labels):

        pass
