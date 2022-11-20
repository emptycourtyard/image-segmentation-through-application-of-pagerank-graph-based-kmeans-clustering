# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 02:44:55 2022

@author: emptycourtyard (nO u3m!3M)
"""

import numpy as np

class Graph():
    
    def __init__(self, arr):
        self.arr = arr
        self.V = len(arr)
        self.centroidDists = [0]
 
    def minDistance(self, dist, sptSet):
        min = 1e7
        min_index = 0
        for v in range(self.V):
            if dist[v] < min and sptSet[v] == False:
                min = dist[v]
                min_index = v
        return min_index
 
    def dijkstra(self, src):
 
        dist = [1e7] * self.V
        dist[src] = 0
        sptSet = [False] * self.V
 
        for cout in range(self.V):
            u = self.minDistance(dist, sptSet)
            sptSet[u] = True
            for v in range(self.V):
                if (self.arr[u][v] > 0 and
                   sptSet[v] == False and
                   dist[v] > dist[u] + self.arr[u][v]):
                       dist[v] = dist[u] + self.arr[u][v]
        return dist
    
    def dijkstraDistanceInitialize(self, centroids):
        self.centroidDists = [[0 for col in range(len(centroids))]
                                 for row in range(self.V)]
        for c in range(len(centroids)):
            dist = self.dijkstra(centroids[c])
            for n in range(self.V):
                self.centroidDists[n][c] = dist[n]
                
    def personalizedPagerank(self, idx, num_iterations: int = 100, d: float = 0.9):
        M = np.array(self.arr)
        N = self.V
        v = np.random.rand(N)
        v = v / np.linalg.norm(v, 1)
        for row in range(N):
            if( np.sum( M[row] ) == 0 ):
                M[row] = np.ones(N)
        M = M / np.matmul(M,np.ones((N,1)))
        p = np.zeros(N)
        p[idx] = (1 - d)
        M_hat =  ((d * M) + p)
        for i in range(num_iterations):
            v = v @ M_hat
        return v
    
    def personalizedPagerankDistanceInitialize(self, centroids):
        self.centroidDists = [[0 for col in range(len(centroids))]
                                 for row in range(self.V)]
        for c in range(len(centroids)):
            dist = self.personalizedPagerank(centroids[c])
            for n in range(self.V):
                self.centroidDists[n][c] = (1 - dist[n]) # highest rank is shortest distance
                
                
    def subgraphPagerank(self, cluster, num_iterations: int = 100, d: float = 0.9):
        M = np.array(self.arr)
        N = self.V
        for row in [i for i in range(N) if i not in cluster]:
            M[row] = np.ones(N)
        v = np.random.rand(N)
        v = v / np.linalg.norm(v, 1)
        for row in range(N):
            if( np.sum( M[row] ) == 0 ):
                M[row] = np.ones(N)
        M = M / np.matmul(M,np.ones((N,1)))
        M_hat = ((d * M) + ((1 - d) / N))
        for i in range(num_iterations):
            v = v @ M_hat
        return v
                
                
    def subgraphPagerankDistanceInitialize(self, sorted_nodes):
        self.centroidDists = [[0 for col in range(len(sorted_nodes))]
                                 for row in range(self.V)]
        for c in range(len(sorted_nodes)):
            cluster = sorted_nodes[c]
            dist = self.subgraphPagerank(cluster)
            for n in range(self.V):
                self.centroidDists[n][c] = (1 - dist[n]) # highest rank is shortest distance
        
        
                
