# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 02:45:35 2022

@author: emptycourtyard (nO u3m!3M)
"""

import numpy as np
import random as random
from graph import Graph

class KMeans:
    
    def __init__(self, n_clusters=3, max_iter=500, graph=Graph([[]])):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.graph = graph
        self.sorted_nodes = [[]]
        
    def fit(self, method: int = 0):
        
        self.centroids = random.sample(range(self.graph.V), self.n_clusters)

        prev_centroids = None
        
        prev_sorted_nodes = [[] for _ in range(self.n_clusters)] # setup previous sorted points (only relevant for subgraphPagerank)
        
        j = 0
        for i in [x for x in range(self.graph.V) if x not in self.centroids]: # uniform distribution replaces code: prev_sorted_points[0] = [i for i in range(self.graph.V) if i not in self.centroids]
            prev_sorted_nodes[j].append(i)
            j += 1
            j = j % self.n_clusters
        for n in range(len(self.centroids)):
            prev_sorted_nodes[n].append(self.centroids[n])
        
        g = np.array(self.graph.arr)
        iteration = 0
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
            
            sorted_nodes = [[] for _ in range(self.n_clusters)]
            
            if(method == 0):
                self.graph.personalizedPagerankDistanceInitialize(self.centroids)
            if(method == 1):
                self.graph.subgraphPagerankDistanceInitialize(prev_sorted_nodes)
            if(method == 2):
                self.graph.dijkstraDistanceInitialize(self.centroids)
            
            for x in range(self.graph.V):
                dists = self.graph.centroidDists[x]
                centroid_idx = np.argmin(dists)
                sorted_nodes[centroid_idx].append(x)
                
            if [] in sorted_nodes: #if empty sorted_nodes cluster exists
                myriad = 0
                empty = []
                for n in range(len(sorted_nodes)):
                    if len(sorted_nodes[n]) == 0:
                        empty.append(n)
                    if len(sorted_nodes[n]) > len(empty):
                        myriad = n
                for n in empty:
                    sorted_nodes[n].append( sorted_nodes[myriad][0] )
                    del sorted_nodes[myriad][0]    
            
            prev_centroids = self.centroids
            prev_sorted_nodes = sorted_nodes
            top_ranks_idxs = [ np.argmax( self.pagerank( g[cluster, :][:, cluster])) for cluster in sorted_nodes]
            self.centroids = [ sorted_nodes[n][top_ranks_idxs[n]] for n in range(len(top_ranks_idxs))] 
            
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():
                    self.centroids[i] = prev_centroids[i]
            iteration += 1
            
        self.sorted_nodes = sorted_nodes
        return self.centroids, sorted_nodes
    
    def pagerank(self, M, num_iterations: int = 100, d: float = 0.9):
        N = M.shape[0]
        if(N == 0):
            return [0]
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
    
    def evaluate(self, idxs):
        sorted_idxs = [[] for row in range(len(self.sorted_nodes))]
        for n in range(len(self.sorted_nodes)):
            for i in self.sorted_nodes[n]:
                sorted_idxs[n].append( idxs[i] )
        return sorted_idxs