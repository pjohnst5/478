from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import numpy as np
from scipy import stats

class Cluster():
    def __init__(self, centroid):
        self.centroid = centroid
        self.instanceIndexes = []
        self.aScores = {}
        self.bScores = {}
        self.sScores = {}
        self.sScore = None

    def __str__(self):
        string = "\nCluster\n"
        string += "centroid: "
        string += str(self.centroid)
        string += "\ninstances tied to this centroid: "
        string += str(self.instanceIndexes)
        string += "\n"
        return string

    def calcSScores(self):
        sum = 0
        for index in self.instanceIndexes:
            bScore = self.bScores[index]
            aScore = self.aScores[index]
            self.sScores[index] = ( ( bScore - aScore ) / ( max(bScore, aScore) ) )
            sum += self.sScores[index]
        self.sScore = sum / len(self.instanceIndexes)


class ClusterLearner(SupervisedLearner):
    """
    This is the cluster Learner
    """

    def __init__(self):
        self.debug = False
        pass

    def arrayToString(self, array):
        string = "["
        for elem in array:
            string += str(round(elem, 3))
            string += ", "
        string += "]"
        return string

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

    def printClusters(self):
        print("\nNumber of clusters:", len(self.clusters))
        for i in range(len(self.clusters)):
            print("\tCentroid", i, "=", self.arrayToString(self.clusters[i].centroid))
        print("Number of instances in each cluster")
        for i in range(len(self.clusters)):
            print("\tCluster",i,": ", len(self.clusters[i].instanceIndexes))
        print("SSEs")
        for i in range(len(self.clusters)):
            print("\tCluster",i,": ", round(self.calcClusterSSE(self.clusters[i]), 3))
        print("Silhouettes")
        for i in range(len(self.clusters)):
            print("\tSilhouette:", round(self.clusters[i].sScore, 3) )
        print("Total SSE: ", round(self.calcTotalSSE(), 3))
        print("Total Silhouette: ", round(self.calcSilhouetteScore(),3))

    def calcClusterSSE(self, cluster):
        if len(cluster.instanceIndexes) == 0:
            raise Exception("Empty cluster error calc Cluster SSE")

        clusterInstances = self.data[cluster.instanceIndexes, :]
        centroid = cluster.centroid

        difference = centroid - clusterInstances

        #if difference is inf, -inf, or nan there was an unknown
        difference = np.where(difference == float("inf"), 1, difference)
        difference = np.where(difference == -float("inf"), 1, difference)
        difference = np.where(np.isnan(difference), 1, difference)

        #fix all nominal columns
        for j in range(self.features.cols):
            #if this attribute is nominal
            if self.features.value_count(j) != 0:
                difference[:, j] = np.where(difference[:, j] != 0, 1, 0)

        #find the euclidean distance for each difference array
        distances = np.linalg.norm(difference, axis=-1)

        #sse is sum of squared euclidean distance of each cluster
        #member to the cluster centroid
        sse = np.sum(distances**2)
        return sse

    def calcTotalSSE(self):
        totalSSE = 0
        for clust in self.clusters:
            totalSSE += self.calcClusterSSE(clust)
        return totalSSE

    def calcDistance(self, centroid, instance):
        difference = centroid - instance

        #if difference is inf, -inf, or nan there was an unknown
        difference = np.where(difference == float("inf"), 1, difference)
        difference = np.where(difference == -float("inf"), 1, difference)
        difference = np.where(np.isnan(difference), 1, difference)

        #fix all nominal columns
        for j in range(self.features.cols):
            #if this attribute is nominal
            if self.features.value_count(j) != 0:
                if difference[j] != 0:
                    difference[j] = 1

        #find the euclidean distance
        distance = np.linalg.norm(difference)
        return distance

    def calcCentroid(self, cluster):
        if len(cluster.instanceIndexes) == 0:
            # raise Exception("Empty cluster error calc centroid")
            return

        clusterInstances = self.data[cluster.instanceIndexes, :]
        newCentroid = np.zeros(self.features.cols, dtype=float)

        for c in range(self.features.cols):
            column = clusterInstances[:, c].copy()

            #if all the values in col are inf, centroid attr val inf
            if np.all(column == float("inf")):
                newCentroid[c] = float("inf")
            else:
                #remove all infinities
                rowIndexesInf = np.argwhere(column == float("inf"))
                column = np.delete(column, rowIndexesInf)

                #if the column is nominal, find most common attr
                if self.features.value_count(c) != 0:
                    newCentroid[c] = stats.mode(column)[0][0]

                #if the column is real, find mean
                else:
                    newCentroid[c] = np.mean(column)
        cluster.centroid = newCentroid

    def findMembers(self):
        #clear old instances from cluster!
        for clust in self.clusters:
            clust.instanceIndexes.clear()

        outString = ""
        if self.debug:
            print("Making assignments")

        for rowIndex in range(self.features.rows):
            shortestDist = float("inf")
            winningClusterIndex = 0
            row = self.data[rowIndex, :]

            for i in range(len(self.clusters)):
                newDist = self.calcDistance(self.clusters[i].centroid, row)
                if newDist < shortestDist:
                    winningClusterIndex = i
                    shortestDist = newDist
            self.clusters[winningClusterIndex].instanceIndexes.append(rowIndex)

            if self.debug:
                if rowIndex % 10 == 0:
                    outString += "\n"
                outString += "\t"
                outString += str(rowIndex)
                outString += "="
                outString += str(winningClusterIndex)

        if self.debug:
            print(outString)


    def calcSilhouetteScore(self):
        #clear old scores
        for clust in self.clusters:
            clust.aScores.clear()
            clust.bScores.clear()
            clust.sScores.clear()

        #calculate a scores
        for clust in self.clusters:

            for index in clust.instanceIndexes:
                distanceSum = 0

                for other in clust.instanceIndexes:
                    if index != other:
                        distanceSum += self.calcDistance(self.data[index], self.data[other])
                clust.aScores[index] = distanceSum/(len(clust.instanceIndexes)-1)

        #calculate b scores
        for clustIndex in range(len(self.clusters)):
            clust = self.clusters[clustIndex]

            for index in clust.instanceIndexes:
                bestBScore = float("inf")

                for otherClustIndex in range(len(self.clusters)):
                    if clustIndex != otherClustIndex:
                        otherClust = self.clusters[otherClustIndex]
                        distanceSum = 0

                        for otherInstIndex in otherClust.instanceIndexes:
                            distanceSum += self.calcDistance(self.data[index], self.data[otherInstIndex])
                        newBScore = distanceSum / len(otherClust.instanceIndexes)
                        if newBScore < bestBScore:
                            bestBScore = newBScore
                clust.bScores[index] = bestBScore

        #now that we have all the a and b scores, find cluster silhouette scores
        for clust in self.clusters:
            clust.calcSScores()

        #now average all cluster silhouette scores of each instance
        sScoreSum = 0
        for clust in self.clusters:
            for index in clust.instanceIndexes:
                sScoreSum += clust.sScores[index]

        return sScoreSum / self.features.rows


    def makeClusters(self, instances, k, randomKs):
        emptyCluster = True
        while emptyCluster:
            emptyCluster = False
            self.features = instances
            self.data = np.array(instances.data)
            self.clusters = []

            if randomKs:
                if self.debug:
                    print("Using random k instances as initial centroids\n")
                # randInstIndexes = np.random.permutation(self.features.rows)[0:k]
                # for index in randInstIndexes:
                #     self.clusters.append(Cluster(self.data[index]))

                minDistance = 1.3
                randInstIndexes = np.random.permutation(self.features.rows)
                chosenCount = 0
                index = 0
                while chosenCount < k:
                    if len(randInstIndexes) - index < (k - chosenCount):
                        raise Exception("Too strict a minimum distance")

                    newInstance = self.data[randInstIndexes[index]]

                    wasTooClose = False
                    for clust in self.clusters:
                        if self.calcDistance(clust.centroid, newInstance) < minDistance:
                            wasTooClose = True

                    if not wasTooClose:
                        self.clusters.append(Cluster(newInstance))
                        chosenCount += 1
                    index += 1

                print("minDist:", minDistance," remaining instances: ", len(randInstIndexes)-index)

            else:
                if self.debug:
                    print("Not using random k instances as initial centroids\n")
                for index in range(k):
                    self.clusters.append(Cluster(self.data[index]))


            oldSSE = float("inf")
            newSSE = 0
            iterationCount = 0
            while(True):
                if self.debug:
                    print("\n*****************")
                    print("Iteration", iterationCount + 1)
                    print("*****************")
                    print("Centroids")

                for i in range(len(self.clusters)):
                    self.calcCentroid(self.clusters[i])

                    if self.debug:
                        print("Centroid", i, "=", self.arrayToString(self.clusters[i].centroid))

                self.findMembers()

                #if there are any empty clusters start over
                for clust in self.clusters:
                    if len(clust.instanceIndexes) == 0:
                        emptyCluster = True
                        break

                if emptyCluster:
                    break
                newSSE = 0
                newSSE = self.calcTotalSSE()
                if self.debug:
                    print("SSE:", newSSE)
                    print("Silhouette Score: ", self.calcSilhouetteScore())

                if newSSE == oldSSE:
                    break
                else:
                    oldSSE = newSSE
                    newSSE = 0
                iterationCount += 1

        self.calcSilhouetteScore()
        self.printClusters()
