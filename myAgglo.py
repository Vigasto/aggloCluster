from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import pairwise_distances
from scipy.spatial import distance
from operator import add
import numpy as np

class MyAgglomerativeClustering(BaseEstimator, ClusterMixin):
    def __init__(self, n_cluster=3, linkage="single"):
        assert n_cluster > 1
        self.n_cluster = n_cluster
        self.linkage = linkage
    def fit(self, X, y=None):
        distance_matrix = pairwise_distances(X)
        self.labels_ = np.zeros(X.shape[0])
        
        #initiate cluster groups
        cluster = []
        for i in range(X.shape[0]):
            init_cluster = [i]
            cluster.append(init_cluster)
        # merges cluster until n_cluster achieved
        while distance_matrix.shape[0] > self.n_cluster:
            # search minimum dissimilarity to merge once
            min_i, min_j = self.get_minimum_idx(distance_matrix)
            # merge cluster members
            for merger in cluster[min_j]:
                cluster[min_i].append(merger)
            del cluster[min_j]
            # renew distance matrix
            distance_matrix = np.delete(distance_matrix, min_j, 0)
            distance_matrix = np.delete(distance_matrix, min_j, 1)
            # recompute distance matrix
            if self.linkage=="single":
                for i in range (0,distance_matrix.shape[0]):
                    if (i==min_i):
                        continue
                    else:
                        minDist = None
                        for clusterItem0 in cluster[min_i]:
                            for clusterItem1 in cluster[i]:
                                if (minDist == None):
                                    minDist = distance.euclidean(X[clusterItem0],X[clusterItem1])
                                elif distance.euclidean(clusterItem0,clusterItem1) < minDist:
                                    minDist = distance.euclidean(X[clusterItem0],X[clusterItem1])
                        distance_matrix[min_i,i] = minDist;
                        distance_matrix[i,min_i] = minDist;
            elif self.linkage=="complete":
                for i in range (0,distance_matrix.shape[0]):
                    if (i==min_i):
                        continue
                    else:
                        maxDist = None
                        for clusterItem0 in cluster[min_i]:
                            for clusterItem1 in cluster[i]:
                                if (maxDist == None):
                                    maxDist = distance.euclidean(X[clusterItem0],X[clusterItem1])
                                elif distance.euclidean(clusterItem0,clusterItem1) > maxDist:
                                    maxDist = distance.euclidean(X[clusterItem0],X[clusterItem1])
                        distance_matrix[min_i,i] = maxDist;
                        distance_matrix[i,min_i] = maxDist;
            elif self.linkage=="average_group":
                for i in range (0,distance_matrix.shape[0]):
                    if (i==min_i):
                        continue
                    else:
                        mean0 = [0]*(X.shape[1])
                        for clusterItem0 in cluster[min_i]:
                            mean0 = list(map(add,mean0,X[clusterItem0]))
                        for element in mean0:
                            element = element / len(cluster[min_i])
                        mean1 = [0]*(X.shape[1])
                        for clusterItem1 in cluster[i]:
                            mean1 = list(map(add,mean1,X[clusterItem0]))
                        for element in mean1:
                            element = element / len(cluster[i])
                        dist = distance.euclidean(mean0,mean1)
                        distance_matrix[min_i,i] = dist;
                        distance_matrix[i,min_i] = dist;
            elif self.linkage=="average":
                for i in range (0,distance_matrix.shape[0]):
                    if (i==min_i):
                        continue
                    else:
                        sumDist = 0;
                        for clusterItem0 in cluster[min_i]:
                            for clusterItem1 in cluster[i]:
                                sumDist += distance.euclidean(X[clusterItem0],X[clusterItem1])
                        meanDist = sumDist / (len(cluster[i]) + len(cluster[min_i]))
                        distance_matrix[min_i,i] = meanDist;
                        distance_matrix[i,min_i] = meanDist;
        #set labels
        i = 0
        for clust in cluster:
            label = i
            for member in clust:
                self.labels_[member] = label
            i = i+1
    def get_minimum_idx(self,m):
        assert (m.shape[0] > 1)
        min = m[1,0]
        min_i = 1;
        min_j = 0;
        for j in range(1, m.shape[0]):
            for i in range(0, j):
                if m[i,j] < min:
                    min = m[i,j]
                    min_i = i
                    min_j = j
        if min_i<min_j:
            return min_i, min_j
        return min_j, min_i