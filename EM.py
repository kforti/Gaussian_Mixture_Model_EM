import numpy as np
import pandas as pd
from scipy.stats import norm
import copy
import math
import time
import matplotlib.pyplot as plt

class Points:
    def __init__(self, point):
        self.point = point
        self.cluster = -1
        self.prob = 0

class Cluster:
    def __init__(self,index,dimension):
        self.label = index
        self.d = dimension
        self.points = []
        self.mu = 0
        self.sigma = 0
        self.count = 0

    def reset(self):
        """
        before every iteration, should set point list to empty
        """
        self.points = []
        self.count = 0

    def compute(self):
        """
        compute mu and sigma of current cluster
        """
        self.count = len(self.points)
        if self.count == 0:
            self.mu = [0.0 for d in range(self.d)]
            self.sigma = [0.0 for d in range(self.d)]
        else:
            self.mu = np.mean(self.points,0)
            self.sigma = np.std(self.points,0)

def readData(filename):
    reader = pd.read_csv(filename)
    reader = np.array(reader)
    data = []
    points = []
    for lines in reader:
        data += [lines]
        points += [Points(lines)]
    data = np.array(data)
    return data, points

def Initial(data, number_clusters, clusters):
    """
    data: (number of data, dimensions)
    number_clusters: (1)
    clusters: list of Cluster
    Randomly select k centers
    """
    mean = np.mean(data,0)   # mean: (dimension)
    std = np.std(data,0)  # std: (dimension)
    centers = []
    for c in range(number_clusters):
        x = np.random.randint(0,len(data)-1)
        while x in centers:
            x = np.random.randint(0, len(data) - 1)
        centers.append(x)
    for num, cluster in enumerate(clusters):
        cluster.mu = data[centers[num]]
        cluster.sigma = std
    return clusters

def GaussianProb(mu, sigma, dimensions, test_value):
    """
    mu, sigma: (number of clusters, number of dimensions), the mus and sigmas of these clusters
    test_value: (number of dimensions), one data point
    return the best fit gaussian of test value
    """
    number_clusters = len(mu)
    dimension = len(mu[0])
    prob = np.array([1 for num in range(number_clusters)]).astype(float)
    for num in range(number_clusters):
        for d in range(dimension):
            prob[num] *= norm.pdf(test_value[d],mu[num][d],sigma[num][d])
    # the following part can be modified, now using soft clustering
    prob = prob / sum(prob)
    x = np.random.rand()
    temp = [0] + list(prob)
    for i in range(1,len(temp)):
        temp[i] = temp[i] + temp[i - 1]
    for c in range(len(temp)-1):
        if x >= temp[c] and x < temp[c + 1]:
            return c, prob[c]
    # cluster = np.argmax(prob)
    # return cluster, prob[cluster]

def Expectation(points, dimensions, clusters):
    """
    mu, sigma: (number of clusters, dimension)
    points: list of class Points
    clusters: (number of clusters)
    get the cluster and probability for points with current parameters
    """
    mu = np.array([cluster.mu for cluster in clusters])
    sigma = np.array([cluster.sigma for cluster in clusters])
    for point in points:
        cluster, prob = GaussianProb(mu, sigma, dimensions, point.point)
        point.cluster = cluster
        point.prob = prob
        clusters[cluster].points.append(point.point)
    # log likelihood?
    return points, clusters

def Maximization(points, number_clusters, dimensions, clusters):
    """
    mu, sigma: (number of clusters, dimension)
    points: list of class Points
    clusters: (number of clusters)
    compute and update the value of mu and sigma for every cluster
    """
    for cluster in clusters:
        cluster.compute()
    return clusters

def EMalgorithm(data, points, number_clusters, iterations, start_time, maxtime):
    clusters = [Cluster(ind,len(data[0])) for ind in range(number_clusters)]
    clusters = Initial(data,number_clusters,clusters)
    for i in range(iterations):
        for cluster in clusters:
            cluster.reset()
        points, clusters = Expectation(points,len(data[0]),clusters)
        clusters = Maximization(points,number_clusters,len(data[0]),clusters)
        time_diff = time.time() - start_time
        if time_diff >= maxtime:   break
    return points, clusters

def fineTuning(data, points, number_clusters, iterations, restart,maxtime):
    start_time = time.time()
    while 1:
        points, clusters = EMalgorithm(data, points, number_clusters, iterations, start_time, maxtime)
        time_diff = time.time() - start_time
        if time_diff < maxtime and restart is True: continue
        else:   break
    print("time: " + str(time_diff))
    return points, clusters

def showClusters(clusters):
    markers = ['bs','go','r^','kx','y.','cv','m*','bo','g*','rx']
    for label,cluster in enumerate(clusters):
        plt.plot(np.array(cluster.points)[:,0],np.array(cluster.points)[:,1],markers[label],markersize=5)
        print("label: "+ str(cluster.label) + ", center: " + str(cluster.mu))
    plt.show()

if __name__ == "__main__":
    data, points = readData("data.csv")
    points, clusters = fineTuning(data,points,3,15,False,10)
    showClusters(clusters)