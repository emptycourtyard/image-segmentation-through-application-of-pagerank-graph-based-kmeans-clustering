#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 02:44:23 2022

@author: emptycourtyard (nO u3m!3M)
"""

import sys
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter

import random as random
import time
import math

import numpy as np
import cv2 as cv

from kmeans import KMeans
from graph import Graph


def createGraph_LabNode_neighborEdge(labimg, bin_size: int = 4):
    
    rows,cols,channels = labimg.shape
    
    idx = []
    
    for i in range(rows):
        for j in range(cols):
            a = int(labimg[i][j][1] / bin_size)
            b = int(labimg[i][j][2] / bin_size)
            color = (a,b)
            if color not in idx: 
                idx.append(color)
    
    graph = [[0 for col in range(len(idx))]
                for row in range(len(idx))]
    
    for i in range(rows):
        for j in range(cols):
            color_list = labimg[i][j]
            a = int(color_list[1] / bin_size)
            b = int(color_list[2] / bin_size)
            color = (a,b)
            if(i != 0): 
                neighbor_color_list = labimg[i-1][j]
                n_a = int(neighbor_color_list[1] / bin_size)
                n_b = int(neighbor_color_list[2] / bin_size)
                neighbor_color = (n_a,n_b)
                graph[idx.index(color)][ idx.index(neighbor_color) ] = 1
            if(j != 0): 
                neighbor_color_list = labimg[i][j-1]
                n_a = int(neighbor_color_list[1] / bin_size)
                n_b = int(neighbor_color_list[2] / bin_size)
                neighbor_color = (n_a,n_b)
                graph[idx.index(color)][ idx.index(neighbor_color) ] = 1
            if(j != cols-1): 
                neighbor_color_list = labimg[i][j+1]
                n_a = int(neighbor_color_list[1] / bin_size)
                n_b = int(neighbor_color_list[2] / bin_size)
                neighbor_color = (n_a,n_b)
                graph[idx.index(color)][ idx.index(neighbor_color) ] = 1
            if(i != rows-1): 
                neighbor_color_list = labimg[i+1][j]
                n_a = int(neighbor_color_list[1] / bin_size)
                n_b = int(neighbor_color_list[2] / bin_size)
                neighbor_color = (n_a,n_b)
                graph[idx.index(color)][ idx.index(neighbor_color) ] = 1
                
    return  idx, graph
        

def createGraph_RGBNode_neighborEdge(bgrimg, bin_size: int = 16):
        
    rows,cols,channels = bgrimg.shape
    idx = []
    
    for i in range(rows):
        for j in range(cols):
            color_list = bgrimg[i][j]
            b = int(color_list[0] / bin_size)
            g = int(color_list[1] / bin_size)
            r = int(color_list[2] / bin_size)
            color = (b,g,r)
            if color not in idx: 
                idx.append(color)
            
    graph = [[0 for col in range(len(idx))]
                for row in range(len(idx))]
    
    for i in range(rows):
        for j in range(cols):
            color_list = bgrimg[i][j]
            b = int(color_list[0] / bin_size)
            g = int(color_list[1] / bin_size)
            r = int(color_list[2] / bin_size)
            color = (b,g,r)
            if(i != 0): 
                neighbor_color_list = bgrimg[i-1][j]
                n_b = int(neighbor_color_list[0] / bin_size)
                n_g = int(neighbor_color_list[1] / bin_size)
                n_r = int(neighbor_color_list[2] / bin_size)
                neighbor_color = (n_b,n_g,n_r)
                graph[idx.index(color)][ idx.index(neighbor_color) ] = 1
            if(j != 0): 
                neighbor_color_list = bgrimg[i][j-1]
                n_b = int(neighbor_color_list[0] / bin_size)
                n_g = int(neighbor_color_list[1] / bin_size)
                n_r = int(neighbor_color_list[2] / bin_size)
                neighbor_color = (n_b,n_g,n_r)
                graph[idx.index(color)][ idx.index(neighbor_color) ] = 1
            if(j != cols-1): 
                neighbor_color_list = bgrimg[i][j+1]
                n_b = int(neighbor_color_list[0] / bin_size)
                n_g = int(neighbor_color_list[1] / bin_size)
                n_r = int(neighbor_color_list[2] / bin_size)
                neighbor_color = (n_b,n_g,n_r)
                graph[idx.index(color)][ idx.index(neighbor_color) ] = 1
            if(i != rows-1): 
                neighbor_color_list = bgrimg[i+1][j]
                n_b = int(neighbor_color_list[0] / bin_size)
                n_g = int(neighbor_color_list[1] / bin_size)
                n_r = int(neighbor_color_list[2] / bin_size)
                neighbor_color = (n_b,n_g,n_r)
                graph[idx.index(color)][ idx.index(neighbor_color) ] = 1
                
    return  idx, graph




def createGraph_hueNode_neighborEdge(hsvimg):

    rows,cols,channels = hsvimg.shape
    idx = []

    for i in range(rows):
        for j in range(cols):
            hue = hsvimg[i][j][0]
            if hue not in idx: 
                idx.append(hue)
            
    graph = [[0 for col in range(len(idx))]
                for row in range(len(idx))]
    
    for i in range(rows):
        for j in range(cols):
            hue = hsvimg[i][j][0]
            if(i != 0): graph[idx.index(hue)][ idx.index(hsvimg[i-1][j][0]) ] = 1
            if(j != 0): graph[idx.index(hue)][ idx.index(hsvimg[i][j-1][0]) ] = 1
            if(j != cols-1): graph[idx.index(hue)][ idx.index(hsvimg[i][j+1][0]) ] = 1
            if(i != rows-1): graph[idx.index(hue)][ idx.index(hsvimg[i+1][j][0]) ] = 1
    
    return  idx, graph


###############################################################################


def createGraph_pixelNode_LabEdge(labimg):
    
    rows,cols,channels = labimg.shape
    idx = []
    
    for i in range(rows):
        for j in range(cols):
            idx.append((i,j))
    
    graph = [[0 for col in range(len(idx))]
                for row in range(len(idx))]
    
    max_iteration = rows
    iterator = 1
    for i in range(rows):
        print(iterator, max_iteration)
        iterator+=1
        for j in range(cols):
            pixel_color = (labimg[i][j][1], labimg[i][j][2])
            for ii in range(rows):
                for jj in range(cols):
                    directed_color = (labimg[ii][jj][1], labimg[ii][jj][2])
                    graph[ idx.index((i, j))][ idx.index( (ii,jj)) ] = print( int( 10 - (math.dist(pixel_color,directed_color)/36)) )
                    
    return idx, graph
              
                   

def createGraph_pixelNode_RGBEdge(bgrimg):
    
    rows,cols,channels = bgrimg.shape
    idx = []
    
    for i in range(rows):
        for j in range(cols):
            idx.append((i,j))
    
    graph = [[0 for col in range(len(idx))]
                for row in range(len(idx))]
    
    
    max_iteration = rows
    iterator = 1
    for i in range(rows):
        print(iterator, max_iteration)
        iterator+=1
        for j in range(cols):
            pixel_color = bgrimg[i][j]
            for ii in range(rows):
                for jj in range(cols):
                    directed_color = bgrimg[ii][jj]
                    graph[idx.index((i, j))][ idx.index( (ii,jj)) ] = ( int( 10 - (math.dist(pixel_color,directed_color)/44)) ) #441.6729559300637 h of 255 cube 
    
    return idx, graph


def createGraph_pixelNode_hueEdge(hsvimg):
    
    rows,cols,channels = hsvimg.shape
    idx = []
    
    for i in range(rows):
        for j in range(cols):
            idx.append((i,j))
            
    graph = [[0 for col in range(len(idx))]
                for row in range(len(idx))]
    
    max_iteration = rows
    iterator = 1
    for i in range(rows):
        print(iterator, max_iteration)
        iterator+=1
        for j in range(cols):
            pixel_hue = hsvimg[i][j][0]
            for ii in range(rows):
                for jj in range(cols):
                    directed_hue = hsvimg[ii][jj][0]
                    graph[idx.index((i, j))][ idx.index( (ii,jj)) ] = ( abs(int((max(((int(pixel_hue) - int(directed_hue)) % 180 ), 
                                                                                     ((int(directed_hue) - int(pixel_hue)) % 180 ))
                                                                                 -90 )
                                                                                 /9  )) )
    return idx, graph
                    



###############################################################################



def mapImage_pixelClusters(img, idx, clusters):
    rows,cols,channels = img.shape
    imgdst = np.zeros([rows,cols,channels],dtype=np.uint8)
    imgdst.fill(0)
    N = len(clusters)
    for n in range(N):
        for p in clusters[n]:
            imgdst[ idx[p][0] ][ idx[p][1] ] = [ int( (180 / N) * n ), 255, 255]
    imgdst = cv.cvtColor(imgdst, cv.COLOR_HSV2BGR)
    return imgdst


def mapImage_hueClusters(hsvimg, idx, clusters):
    rows,cols,channels = hsvimg.shape
    imgdst = np.zeros([rows,cols,channels],dtype=np.uint8)
    imgdst.fill(0)
    N = len(clusters)
    for i in range(rows):
        for j in range(cols):
            hue = hsvimg[i][j][0]
            for n in range(N):
                if( idx.index(hue) in clusters[n]):
                    imgdst[i][j] = [ int( (180 / N) * n ), 255, 255]
    imgdst = cv.cvtColor(imgdst, cv.COLOR_HSV2BGR)
    return imgdst


def mapImage_RGBClusters(bgrimg, idx, clusters, bin_size: int = 16):
    rows,cols,channels = bgrimg.shape
    imgdst = np.zeros([rows,cols,channels],dtype=np.uint8)
    imgdst.fill(0)
    N = len(clusters)
    for i in range(rows):
        for j in range(cols):
            color_list = bgrimg[i][j]
            b = int(color_list[0] / bin_size)
            g = int(color_list[1] / bin_size)
            r = int(color_list[2] / bin_size)
            color = (b,g,r)
            if color in idx:
                for n in range(N):
                    if( idx.index(color) in clusters[n]):
                        imgdst[i][j] = [ int( (180 / N) * n ), 255, 255]
            else:
                print(color, 'not in list')
    imgdst = cv.cvtColor(imgdst, cv.COLOR_HSV2BGR)
    return imgdst
    

def mapImage_LabClusters(labimg, idx, clusters, bin_size: int = 4):
    rows,cols,channels = labimg.shape
    imgdst = np.zeros([rows,cols,channels],dtype=np.uint8)
    imgdst.fill(0)
    N = len(clusters)
    for i in range(rows):
        for j in range(cols):
            color_list = labimg[i][j]
            a = int(color_list[1] / bin_size)
            b = int(color_list[2] / bin_size)
            color = (a,b)
            for n in range(N):
                if( idx.index(color) in clusters[n]):
                    imgdst[i][j] = [ int( (180 / N) * n ), 255, 255]
    imgdst = cv.cvtColor(imgdst, cv.COLOR_HSV2BGR)
    return imgdst




def process(args):
            
    graph_type = None
    
    if args.graph_type.isnumeric():
        graph_type = int(args.graph_type)
    else:
        if args.graph_type == 'rgbNode':
            graph_type = 0
        if args.graph_type == 'hueNode':
            graph_type = 1
        if args.graph_type == 'abNode':
            graph_type = 2
        if args.graph_type == 'rgbEdge':
            graph_type = 3
        if args.graph_type == 'hueEdge':
            graph_type = 4
        if args.graph_type == 'abEdge':
            graph_type = 5
            
    distance_algorithm = None
        
    if args.distance_algorithm.isnumeric():
        distance_algorithm = int(args.distance_algorithm)
    else:
        if args.distance_algorithm == 'perPagerank':
            graph_type = 0
        if args.distance_algorithm == 'undPagerank':
            graph_type = 1
        if args.distance_algorithm == 'dijkstra':
            graph_type = 2
            
    try:
        st = time.time()
        
        print('ok2')
        if graph_type == 0: #'rgbNode'
            img = cv.imread(args.input)
            idx, graph = createGraph_RGBNode_neighborEdge(img, bin_size = args.bin_size)
            g = Graph(graph)
            kmeans = KMeans(n_clusters=args.clusters, graph=g)
            centroids, clusters = kmeans.fit(distance_algorithm)
            dstimg = mapImage_RGBClusters(img, idx, clusters, bin_size = args.bin_size)
            cv.imwrite(args.output,dstimg)
            
            
        if graph_type == 1: #'hueNode'
            img = cv.imread(args.input)
            img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            idx, graph = createGraph_hueNode_neighborEdge(img)
            g = Graph(graph)
            kmeans = KMeans(n_clusters=args.clusters, graph=g)
            centroids, clusters = kmeans.fit(distance_algorithm)
            dstimg = mapImage_hueClusters(img, idx, clusters)
            cv.imwrite(args.output,dstimg)
            
        
            
        if graph_type == 2: #'abNode'
            img = cv.imread(args.input)
            img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
            idx, graph = createGraph_LabNode_neighborEdge(img, bin_size = args.bin_size)
            g = Graph(graph)
            kmeans = KMeans(n_clusters=args.clusters, graph=g)
            centroids, clusters = kmeans.fit(distance_algorithm)
            dstimg = mapImage_LabClusters(img, idx, clusters, bin_size = args.bin_size)
            cv.imwrite(args.output,dstimg)
            
            
        if graph_type == 3: #'rgbEdge'
            img = cv.imread(args.input)
            idx, graph = createGraph_pixelNode_RGBEdge(img)
            g = Graph(graph)
            kmeans = KMeans(n_clusters=args.clusters, graph=g)
            centroids, clusters = kmeans.fit(distance_algorithm)
            dstimg = mapImage_pixelClusters(img, idx, clusters)
            cv.imwrite(args.output,dstimg)
            
            
        if graph_type == 4: #'hueEdge'
            img = cv.imread(args.input)
            img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            idx, graph = createGraph_pixelNode_hueEdge(img)
            g = Graph(graph)
            kmeans = KMeans(n_clusters=args.clusters, graph=g)
            centroids, clusters = kmeans.fit(distance_algorithm)
            dstimg = mapImage_pixelClusters(img, idx, clusters)
            cv.imwrite(args.output,dstimg)
            
            
        if graph_type == 5: #'abEdge'
            img = cv.imread(args.input)
            img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
            idx, graph = createGraph_pixelNode_LabEdge(img)
            g = Graph(graph)
            kmeans = KMeans(n_clusters=args.clusters, graph=g)
            centroids, clusters = kmeans.fit(distance_algorithm)
            dstimg = mapImage_pixelClusters(img, idx, clusters)
            cv.imwrite(args.output,dstimg)
        
        et = time.time()
        elapsed_time = et - st
        print('Seconds elapsed:', elapsed_time)
        
    except:
        print("An error occurred. Check argumnets for any mistakes.")

    
    

def main():
    parser = ArgumentParser("segment",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    
    parser.add_argument('--input', required=True, type = str,
                        help='Input image file (supported by opencv) (string)')
      
    parser.add_argument('--output', required=True, type = str,
                        help='Output image file (supported by opencv) (string) e.x. out.PNG')
    
    parser.add_argument('--clusters', required=True, default=3, type=int,
                        help='Number of clusters (integer)')
    
    parser.add_argument('--graph_type', required=True, default=0,
                        help='Graph representation (integer or string) 0:"rgbNode" 1:"hueNode" 2:"abNode" 3:"rgbEdge" 4:"hueEdge" 5:"abEdge"')
    
                        #TODO
                        #help='Graph representation (integer or string) 0:"rgbNode" 1:"hueNode" 2:"abNode" 3:"rgbEdge" 4:"hueEdge" 5:"abEdge 6:"ycrcbNode" 7:"hsvNode" 8:"labNode" 9:"ycrcEdge" 10:"hsvEdge" 11:"labEdge"')
    
    parser.add_argument('--distance_algorithm', required=True, default=0,
                        help='PageRank distance algorithm (integer or string) 0:"perPagerank" 1:"undPagerank" 2:"dijkstra"')
    
    parser.add_argument('--bin_size', required=False, default=4, type=int,
                        help='Optional color intensity reduction applicable for graph_type(s) 0,1, and 2, (integer) e.x. 2 reduces 256 color values to 128')
    
    args = parser.parse_args()
    
    process(args)
    
if __name__ == "__main__":
    sys.exit(main())
  