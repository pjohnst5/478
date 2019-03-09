from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix

import numpy as np
from math import log
import random as rand

class Node():
    def __init__(self):
        self.nodeID = None
        self.children = set()
        self.dataFeat = None
        self.dataLabels = None
        self.allOutputClasses = None
        self.remainingAttrs = None
        self.connectingAttr = None
        self.connectingVal = None
        self.depth = None

    def isLeafNode(self):
        if len(self.children) == 0:
            return True
        else:
            return False

    def baseInfo(self):
        numRemainingInstances = len(self.dataFeat)
        outClasses, counts = np.unique(self.dataLabels, return_counts=True)
        baseInfo = 0
        for count in counts:
            ratio = (count/numRemainingInstances)
            baseInfo += ((-ratio) * log(ratio, 2))
        return baseInfo

    def canGoFurther(self):
        if len(self.remainingAttrs) == 0:
            return False
        unique = np.unique(self.dataLabels)
        if len(unique) == 1:
            return False
        return True

    def majorityClass(self):
        unique, counts = np.unique(self.dataLabels, return_counts=True)
        maxIndexes = np.argmax(counts)
        if isinstance(maxIndexes, np.int64):
            return unique[maxIndexes]
        return unqiue[maxIndexes[0]]

    def __str__(self):
        string = "\nNode ID: "
        string += str(self.nodeID)
        string += "\nNum Children: "
        string += str(len(self.children))
        string += "\nNum remaining instances: "
        string += str(len(self.dataFeat))
        string += "\nRemaining attrs: "
        for attr in self.remainingAttrs:
            string += str(attr)
            string += " "
        string += "\nConnecting attr: "
        string += str(self.connectingAttr)
        string += "\nConnecting val: "
        string += str(self.connectingVal)
        string += "\nDepth: "
        string += str(self.depth)
        string += "\nIs Leaf Node: "
        string += str(self.isLeafNode())
        string += "\nAll output classes: "
        for out in self.allOutputClasses:
            string += str(out)
            string += "  "
        string += "\nOutput class, count : "
        outClasses, counts = np.unique(self.dataLabels, return_counts=True)
        for i in range(len(outClasses)):
            string += "("
            string += str(outClasses[i])
            string += ", "
            string += str(counts[i])
            string += ") "
        string += "\nCan go further: "
        string += str(self.canGoFurther())

        string += "\nBase Info: "
        string += str(self.baseInfo())
        string += "\n"
        return string






class DecisionTreeLearner(SupervisedLearner):
    """
    This is the Decision Tree Learner
    """
    def __init__(self):
        self.debug = False
        self.nodeID = 0
        self.greatestDepth = 0
        self.nodesTakenOut = 0
        self.newGreatestDepth = 0
        pass

    def printTree(self, node):
        print(node)
        for child in node.children:
            print(child)

    def calculateDepth(self, node):
        if node.depth > self.newGreatestDepth:
            self.newGreatestDepth = node.depth
        for child in node.children:
            self.calculateDepth(child)

    def prune(self, node):
        if node.isLeafNode():
            return

        for child in node.children:
            self.prune(child)

        mseBefore = self.calculateMSE()
        temp = set(node.children)
        node.children = set()
        mseAfter = self.calculateMSE()

        if mseAfter > mseBefore:
            node.children = temp
        else:
            self.nodesTakenOut += len(temp)


    def calculateMSE(self):
        pred = []
        sse = 0.0
        for i in range(self.validFeatures.rows):
            feat = self.validFeatures.row(i)
            targ = self.validLabels.row(i)
            self.predict(feat, pred)
            delta = targ[0] - pred[0]
            sse += delta**2
        return sse / self.validFeatures.rows

    def createSets(self, features, labels, percent):
        features.shuffle(labels)
        rowCountTrain = int(features.rows * (1-percent))
        self.trainFeatures = Matrix(features, 0, 0, rowCountTrain, features.cols)
        self.validFeatures = Matrix(features, rowCountTrain, 0, features.rows-rowCountTrain, features.cols)
        self.trainLabels = Matrix(labels, 0, 0, rowCountTrain, labels.cols)
        self.validLabels = Matrix(labels, rowCountTrain, 0, features.rows-rowCountTrain, features.cols)


    def splitNodeOnAttr(self, node, attr):
        attrColFeat = node.dataFeat[:,attr]
        uniqueAttrVals = np.unique(attrColFeat)
        #for every unique value of this attrubite
        for uniqueVal in uniqueAttrVals:
            indicesOfVal = np.where(attrColFeat == uniqueVal)
            featsWithUniqueVal = node.dataFeat[indicesOfVal]
            labelsWithUniqueVal = node.dataLabels[indicesOfVal]
            #make a new node
            child = Node()
            child.nodeID = self.nodeID
            self.nodeID += 1
            child.dataFeat = featsWithUniqueVal
            child.dataLabels = labelsWithUniqueVal
            child.allOutputClasses = node.allOutputClasses
            child.remainingAttrs = np.delete(node.remainingAttrs, np.where(node.remainingAttrs == attr))
            child.connectingAttr = attr
            child.connectingVal = uniqueVal
            child.depth = node.depth + 1
            if child.depth > self.greatestDepth:
                self.greatestDepth = child.depth

            #add this new node as a child of first one
            node.children.add(child)

    def calcAttrInfo(self, attr, dataFeat, dataLabels):
        attrColFeat = dataFeat[:,attr]
        numInstances = len(attrColFeat)
        uniqueAttrVals, countsAttr = np.unique(attrColFeat, return_counts=True)
        outsideSum = 0

        for i in range(len(uniqueAttrVals)):
            outsideRatio = countsAttr[i] / numInstances
            #see how many of these four are in each output class
            indices = np.where(attrColFeat == uniqueAttrVals[i])
            labels = dataLabels[indices]
            #of these labels, do a check for each output class
            uniqueOutputs, countsOutputs = np.unique(labels, return_counts=True)
            insideSum = 0
            for j in range(len(uniqueOutputs)):
                insideRatio = countsOutputs[j] / countsAttr[i]
                insideSum += ((-insideRatio) * log(insideRatio,2))
            outsideSum += (outsideRatio * insideSum)
        return outsideSum


    def exploreNode(self, node):
        #is node pure or out of attributes? If so, done
        if not node.canGoFurther():
            return
        #find info gain for this current node
        baseInfo = node.baseInfo()
        bestAttr = -1
        bestAttrInfoGain = -1
        #for each remaining attribute calculate the info gain
        for attr in node.remainingAttrs:
            #calculate the info gain from splitting on this attribute
            attrInfo = self.calcAttrInfo(attr, node.dataFeat, node.dataLabels)
            infoGain = baseInfo - attrInfo
            if self.debug:
                print("attr: ", attr)
                print("baseInfo: ", baseInfo)
                print("attrInfo: ", attrInfo)
                print("infoGain: ", infoGain,"\n")

            if infoGain > bestAttrInfoGain:
                bestAttrInfoGain = infoGain
                bestAttr = attr

        #Split on best attribute
        bestAttr = rand.choice(node.remainingAttrs)
        self.splitNodeOnAttr(node, bestAttr)

        if self.debug:
            print("best attr: ", bestAttr)
            print("best attr infoGain: ", bestAttrInfoGain, "\n")
            print("After split:")
            print(node)
            for child in node.children:
                print(child)

        for child in node.children:
            self.exploreNode(child)
        return

    def train(self, features, labels):
        #seperate out validation set
        self.createSets(features, labels, 0.2)

        self.root = Node()
        self.root.nodeID = self.nodeID
        self.nodeID += 1
        self.root.dataFeat = np.array(self.trainFeatures.data)
        self.root.dataLabels = np.array(self.trainLabels.data)
        self.root.allOutputClasses = np.unique(labels.data)
        self.root.remainingAttrs = np.arange(len(features.data[0]))
        self.root.depth = 0

        if self.debug:
            print(self.root)
        self.exploreNode(self.root)

        #calculate mse before pruning
        mseBefore = self.calculateMSE()
        print("\nNum nodes before: ", self.nodeID)
        print("Depth before: ", self.greatestDepth)
        print("Mse before: ", mseBefore)
        #prune
        self.prune(self.root)
        self.calculateDepth(self.root)
        #calculate mse after pruning
        mseAfter = self.calculateMSE()
        print("\nNum nodes after: ", self.nodeID - self.nodesTakenOut)
        print("Depth after: ", self.newGreatestDepth)
        print("Mse after: ", mseAfter, "\n")

    def predictDescent(self, node, features):
        #if leaf node, return the majority class
        if node.isLeafNode():
            return node.majorityClass()

        #otherwise, descend down the correct child
        for child in node.children:
            if child.connectingVal == features[child.connectingAttr]:
                return self.predictDescent(child, features)
        #if it gets out here, it means none of the children had this next attribute
        #that means it's the majority class of node
        return node.majorityClass()

    def predict(self, features, labels):
        del labels[:]
        prediction = self.predictDescent(self.root, features)
        labels.append(prediction)
