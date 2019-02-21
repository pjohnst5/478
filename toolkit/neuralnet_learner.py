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
        features.shuffle(labels)
        rowCountTrain = int(features.rows * (1-percent))

        if vowel:
            self.trainFeatures = Matrix(features, 0, 3, rowCountTrain, features.cols - 3)
            self.validFeatures = Matrix(features, rowCountTrain, 3, features.rows-rowCountTrain, features.cols - 3)
        else:
            self.trainFeatures = Matrix(features, 0, 0, rowCountTrain, features.cols)
            self.validFeatures = Matrix(features, rowCountTrain, 0, features.rows-rowCountTrain, features.cols)
        self.trainLabels = Matrix(labels, 0, 0, rowCountTrain, labels.cols)
        self.validLabels = Matrix(labels, rowCountTrain, 0, features.rows-rowCountTrain, features.cols)

        # add a bias column to training features, make it 1
        self.trainInputs = np.ones((self.trainFeatures.rows, self.trainFeatures.cols + 1), dtype=float)
        self.trainInputs[:,:-1] = self.trainFeatures.data
        # add a bias colunn to validation features, make it 1
        self.validInputs = np.ones((self.validFeatures.rows, self.validFeatures.cols + 1), dtype=float)
        self.validInputs[:,:-1] = self.validFeatures.data


    #the first int in nodes is number of output nodes
    #the last int in nodes is the number of input nodes
    def createNetwork(self, nodes):
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

    def shuffleDecks(self):
        #shuffle training deck
        self.trainFeatures.shuffle(self.trainLabels)
        self.trainInputs[:,:-1] = self.trainFeatures.data
        #shuffle validation deck
        self.validFeatures.shuffle(self.validLabels)
        self.validInputs[:,:-1] = self.validFeatures.data

    def checkValidation(self):
        sseValid = 0.0
        mseValid = 0.0
        classificatonAccuracyVS = 0.0
        numCorrect = 0
        numTotal = len(self.validInputs)
        outputIndexes = self.nodeIndexes[0]

        #for each validation input
        for valIndex in range(len(self.validInputs)):
            #get the validation input
            valInput = self.validInputs[valIndex].tolist()
            #get what the answer is supposed to be
            label = int(self.validLabels.row(valIndex)[0])
            #seng input through the network and also predict
            labels = []
            greatestOutputIndex = self.predict(valInput[:-1], labels)

            #Set up target array
            targets = np.zeros(len(outputIndexes))
            #If it's continuous, keep the label as is
            if len(targets) == 1:
                targets[0] = label
            #if it's nominal, set the node who is supposed to say yes to 1, all others 0
            else:
                targets[int(label)] = 1

            #get all the outputs
            outputs = []
            for out in outputIndexes:
                outputs.append(self.output[out])

            #its a continuous response variable
            if len(outputIndexes) == 1:
                #check if the output was correct
                if self.output[greatestOutputIndex] == label:
                    numCorrect += 1
            #if it's a nominal response variable
            else:
                #check if the output was correct
                if greatestOutputIndex == label:
                    numCorrect += 1

            #get the difference list
            differences = []
            for out in outputIndexes:
                differences.append(targets[out] - outputs[out])
            #square the differences
            differencesSquared = []
            for diff in differences:
                differencesSquared.append(diff**2)
            #sum up the squared differences list
            sseValid += sum(differencesSquared)

        #calculate the %classified correctly
        classificatonAccuracyVS = numCorrect / numTotal

        #calculate MSE of validation set
        mseValid = sseValid / numTotal
        return mseValid, classificatonAccuracyVS

    def trainingSSE(self, input, label):
        outputIndexes = self.nodeIndexes[0]
        #seng input through the network and also predict
        labels = []
        greatestOutputIndex = self.predict(input[:-1], labels)

        #Set up target array
        targets = np.zeros(len(outputIndexes))
        #If it's continuous, keep the label as is
        if len(targets) == 1:
            targets[0] = label
        #if it's nominal, set the node who is supposed to say yes to 1, all others 0
        else:
            targets[int(label)] = 1

        #get all the outputs
        outputs = []
        for out in outputIndexes:
            outputs.append(self.output[out])

        #get the difference list
        differences = []
        for out in outputIndexes:
            differences.append(targets[out] - outputs[out])
        #square the differences
        differencesSquared = []
        for diff in differences:
            differencesSquared.append(diff**2)
        #sum up the squared differences list
        sseIncrement = sum(differencesSquared)

        return sseIncrement


    def forwardProp(self, input):
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



    def train(self, features, labels):
        self.learningRate = 0.085
        self.momentum = 0.125
        #seperate out training, testing and validation sets
        #third argument is percent for validation, last is if vowel dataset
        self.createSets(features, labels, 0.2, True)

        inputNodeCount = self.trainFeatures.cols+1
        outputNodeCount = 0
        #if the value count is 0, it's continuous, 1 output node
        if labels.value_count(0) == 0:
            outputNodeCount = 1
        #else there are multiple classes, make that many output nodes
        else:
            outputNodeCount = labels.value_count(0)

        #First num in array is num of output nodes, last is num of input nodes
        #numbers between are node counts for hidden layers (including bias)
        self.createNetwork([outputNodeCount, inputNodeCount - 1, (inputNodeCount * 2) - 1, inputNodeCount])

        done = False
        inputIndex = 0
        totalEpochs = 0
        epochsNoProgress = 0
        epochsNoProgressCap = 5
        sseTrain = 0.0
        mseTrain = 0.0
        mseValid = 0.0
        bssfMSE = float("inf")
        epochsIndexes = []
        mseTrains = []
        mseValids = []
        validationAccuracies = []

        while not done:
            singleInput = self.trainInputs[inputIndex]
            label = self.trainLabels.row(inputIndex)[0]

            sseIncrement = self.trainingSSE(singleInput.tolist(), label)
            sseTrain += sseIncrement
            self.backProp(label)

            inputIndex += 1

            #If at the end of an epoch
            if inputIndex == len(self.trainInputs):
                inputIndex = 0
                totalEpochs += 1
                #shuffle training and validation decks
                self.shuffleDecks()

                #calculate training MSE
                mseTrain = sseTrain / len(self.trainInputs)
                #push it on the train MSEs
                epochsIndexes.append(totalEpochs-1)
                mseTrains.append(mseTrain)
                mseTrain = 0.0
                sseTrain = 0.0

                #check validation set
                mseValid, classificatonAccuracyVS = self.checkValidation()
                mseValids.append(mseValid)
                validationAccuracies.append(classificatonAccuracyVS)

                #if this mseValid is better than the bssfMse
                if mseValid < bssfMSE:
                    epochsNoProgress = 0
                    bssfMSE = mseValid
                #if not, increment epochs with no progress
                else:
                    epochsNoProgress += 1

                #if we've reached the cap of epochs with no progress
                if epochsNoProgress == epochsNoProgressCap:
                    done = True
        # print("epochsIndexes = ", epochsIndexes)
        # print("mseTrains = ", mseTrains)
        # print("mseValids = ", mseValids)
        # print("validationAccuracies = ", validationAccuracies)
        # print("Best VS MSE: ", bssfMSE)
        # print("Training MSE: " , mseTrains[-1])
        # print("total epochs: ", totalEpochs)


    def predict(self, features, labels):
        del labels[:]

        #add a bias weight onto the input
        features.append(1.0)

        #run the input through the network
        self.forwardProp(features)

        #grab the output node indexes
        outputNodeIndexes = self.nodeIndexes[0]
        #make the best output index the first output node
        bestOutputIndex = outputNodeIndexes[0]
        prediction = 0

        #if output class is continuous
        if len(outputNodeIndexes) == 1:
            prediction = self.output[bestOutputIndex]
        #if the output class is nominal
        else:
            #find the output node index with the highest output
            for out in outputNodeIndexes:
                if self.output[out] > self.output[bestOutputIndex]:
                    bestOutputIndex = out
            prediction = bestOutputIndex

        #put this prediction as the label
        if len(labels) == 0:
            labels.append(prediction)
        else:
            labels[0] = prediction

        return bestOutputIndex


    def accuracy_and_mse(self, features, labels, confusion=None):
        mse = 0.0
        sse = 0.0
        outputs = []
        targets = []
        differences = []
        differencesSquared = []

        if features.rows != labels.rows:
            raise Exception("Expected the features and labels to have the same number of rows")
        if labels.cols != 1:
            raise Exception("Sorry, this method currently only supports one-dimensional labels")
        if features.rows == 0:
            raise Exception("Expected at least one row")

        label_values_count = labels.value_count(0)
        if label_values_count == 0:
            # label is continuous
            pred = []
            sse = 0.0
            for i in range(features.rows):
                feat = features.row(i)
                targ = labels.row(i)
                pred[0] = 0.0       # make sure the prediction is not biased by a previous prediction
                self.predict(feat, pred)
                delta = targ[0] - pred[0]
                sse += delta**2
            return math.sqrt(sse / features.rows)

        else:
            # label is nominal, so measure predictive accuracy
            if confusion:
                confusion.set_size(label_values_count, label_values_count)
                confusion.attr_names = [labels.attr_value(0, i) for i in range(label_values_count)]

            correct_count = 0
            prediction = []
            for i in range(features.rows):
                feat = features.row(i)
                targ = int(labels.get(i, 0))
                if targ >= label_values_count:
                    raise Exception("The label is out of range")
                self.predict(feat, prediction)
                pred = int(prediction[0])
                #print("pred: ", pred, "\n")
                if confusion:
                    confusion.set(targ, pred, confusion.get(targ, pred)+1)


    #----------------  calculate differences, differences squared --------------#
                outputIndexes = self.nodeIndexes[0]
                #send input through the network and also predict
                labelz = []
                greatestOutputIndex = self.predict(feat, labelz)

                #Set up target array
                targets = np.zeros(len(outputIndexes))
                #If it's continuous, keep the label as is
                if len(targets) == 1:
                    targets[0] = targ
                #if it's nominal, set the node who is supposed to say yes to 1, all others 0
                else:
                    targets[int(targ)] = 1

                #get all the outputs
                outputs = []
                for out in outputIndexes:
                    outputs.append(self.output[out])

                #get the difference list
                differences = []
                for out in outputIndexes:
                    differences.append(targets[out] - outputs[out])
                #square the differences
                differencesSquared = []
                for diff in differences:
                    differencesSquared.append(diff**2)
                #sum up the squared differences list
                sseIncrement = sum(differencesSquared)
                sse += sseIncrement
#------------------------------------------------------------------



                if pred == targ:
                    correct_count += 1
                mse = sse / features.rows
            return (correct_count / features.rows), mse
