# CSCI 323 Winter
# Assignment 4
# Fariha Khan

import sys
import time
from sys import maxsize
from collections import defaultdict
from os import listdir
from os.path import isfile, join
import math


# open the file
def read_graph(file_name):
    # use with open to open the file.
    with open(file_name, 'r') as file:
        graph = []
        lines = file.readlines()
        for line in lines:
            costs = line.split(' ')
            row = []
            for cost in costs:
                row.append(int(cost))
            graph.append(row)
        return graph


def desc_graph(graph):
    num_vertices = len(graph)
    message = ''
    message += 'Number of vertices = ' + str(num_vertices) + '\n'
    non_zero = 0
    for i in range(num_vertices):
        for j in range(num_vertices):
            if graph[i][j] > 0:
                non_zero += 1
    num_edges = int(non_zero / 2)
    message += 'Number of edges = ' + str(num_edges) + '\n'
    message += 'Symmetric = ' + str(is_symmetric(graph)) + '\n'
    return message


def is_symmetric(graph):
    num_vertices = len(graph)
    for i in range(num_vertices):
        for j in range(num_vertices):
            if graph[i][j] != graph[j][i]:
                return False
    return True


def print_graph(graph, sep=' '):
    str_graph = ''
    for row in range(len(graph)):
        str_graph += sep.join([str(c) for c in graph[row]]) + '\n'
    return str_graph


def analyze_graph(file_name):
    graph = read_graph(file_name)
    output_file_name = file_name[0:-4 + len(file_name)] + '_report.txt'
    with open(output_file_name, 'w') as output_file:
        output_file.write('Analysis of graph: ' + file_name + '\n\n')
        str_graph = print_graph(graph)
        output_file.write(str_graph + '\n')
        graph_descrip = desc_graph(graph)
        output_file.write(graph_descrip + '\n')
        dfs_traversal = dfs(graph)
        bfs_traversal = bfs(graph)
        prim_traversal = primMST(graph)
        kruskal_traversal = kruskalMST(graph)
        dij_traversal = dijkstra_sssp(graph)
        floyd_traversal = floyd_asps(graph)
        output_file.write('dfs traversal: ' + str(dfs_traversal) + '\n')
        output_file.write('bfs traversal: ' + str(bfs_traversal) + '\n')
        output_file.write('Prim traversal: ' + str(prim_traversal) + '\n')
        output_file.write('Kruskal : ' + str(kruskal_traversal) + '\n')
        output_file.write('Floyd traversal: ' + str(floyd_traversal) + '\n')
        output_file.write('dij traversal: ' + str(dij_traversal) + '\n')


# Depth-First Search.
# Code Provided in class
def dfs_util(graph, v, visited):
    visited.append(v)
    for col in range(len(graph[v])):
        if graph[v][col] > 0 and col not in visited:
            dfs_util(graph, col, visited)


def dfs(graph):
    start = time.time()
    visited = []
    dfs_util(graph, 0, visited)
    end = time.time()
    total = end - start
    print(f"DFS Runtime of the program is {total * 10000}")

    return visited


# Breadth-First Search.
# https://stackoverflow.com/questions/43375515/breadth-first-search-with-adjacency-matrix
def bfs_wrapper(graph, u, v, visited):
    nodes = [(u, v)]
    while nodes:
        u, v = nodes.pop(0)
        # the below conditional ensures that our algorithm
        # stays within the bounds of our matrix.
        if u >= len(graph) or v >= len(graph[0]) or u < 0 or v < 0:
            continue
        if (u, v) not in visited:
            if graph[u][v] == 1:
                visited.append((u, v))
                nodes.append((u + 1, v))
                nodes.append((u, v + 1))
                nodes.append((u - 1, v))
                nodes.append((u, v - 1))


def bfs(graph):
    start = time.time()
    visited = []
    for i in range(len(graph)):
        for j in range(len(graph[0])):
            if (i, j) not in visited:
                bfs_wrapper(graph, i, j, visited)
    end = time.time()
    total = end - start
    print(f"BFS Runtime of the program is {total * 10000}")

    return visited


# Prim's MST algorithm.
# https://www.geeksforgeeks.org/prims-algorithm-simple-implementation-for-adjacency-matrix
# -representation/# Function to construct and print MST for a graph
INT_MAX = maxsize


# Returns true if edge u-v is a valid edge to be
# includes in MST. An edge is valid if one end is
# already included in MST and other is not in MST.
def isValidEdge(u, v, inMST):
    if u == v:
        return False
    if inMST[u] == False and inMST[v] == False:
        return False
    elif inMST[u] == True and inMST[v] == True:
        return False
    return True


def primMST(graph):
    start = time.time()
    num_vertices = len(graph)
    inMST = [False] * num_vertices

    # Include first vertex in MST
    inMST[0] = True

    # Keep adding edges while number of included
    # edges does not become V-1.
    edge_count = 0
    mincost = 0
    while edge_count < num_vertices - 1:

        # Find minimum weight valid edge.
        min = INT_MAX
        a = -1
        b = -1
        for i in range(num_vertices):
            for j in range(num_vertices):
                if graph[i][j] < min:
                    if isValidEdge(i, j, inMST):
                        min = graph[i][j]
                        a = i
                        b = j
        if a != -1 and b != -1:
            print("Prim' Algorithm Edge %d: (%d, %d) cost: %d" %
                  (edge_count, a, b, min))
            edge_count += 1
            mincost += min
            inMST[b] = inMST[a] = True
    end = time.time()
    print(f"Prim Algo Runtime of the program is {(end - start) * 10000}")
    print("Cost = %d" % mincost)

    # return primMST(graph)


# Kruskal's algorithm
# Find set of vertex i
def find(parent, i):
    if parent[i] == i:
        return i
    return find(parent, parent[i])


# Does union of i and j. It returns
# false if i and j are already in same set.
def union(i, j, parent):
    a = find(parent, i)
    b = find(parent, j)
    parent[a] = b


# Finds MST using Kruskal's algorithm
def kruskalMST(graph):
    start = time.time()
    parent = []
    num_vertices = len(graph)
    min_cost = 0  # Cost of min MST
    for i in range(num_vertices):
        parent = [i for i in range(num_vertices)]

    # Include minimum weight edges one by one
    e = 0
    while e < num_vertices - 1:
        min = math.inf
        a = -1
        b = -1
        for i in range(num_vertices):
            for j in range(num_vertices):
                if find(parent, i) != find(parent, j) and 0 < graph[i][j] < min:
                    min = graph[i][j]
                    a = i
                    b = j
        union(a, b, parent)
        print('Kruskal Algorithm Edge {}:({}, {}) cost:{}'.format(e, a, b, min))
        e += 1

    print("Minimum cost= {}".format(min_cost))
    end = time.time()
    print(f"Kruskal Algorithm Runtime of the program is {(end - start) * 10000}")

    # return kruskalMST(graph)


# Function that implements Dijkstra's single source
# the shortest path algorithm for a graph represented
# using adjacency matrix representation
def dijkstra_sssp(graph, src):
    num_vertices = len(graph)

    dist = [sys.maxsize] * num_vertices
    dist[src] = 0
    sptSet = [False] * num_vertices

    for cout in range(num_vertices):

        # Pick the minimum distance vertex from
        # the set of vertices not yet processed.
        # u is always equal to src in first iteration
        u = graph.minDistance(dist, sptSet)

        # Put the minimum distance vertex in the
        # shortest path tree
        sptSet[u] = True

        # Update dist value of the adjacent vertices
        # of the picked vertex only if the current
        # distance is greater than new distance and
        # the vertex in not in the shortest path tree
        for v in range(num_vertices):
            if graph[u][v] > 0 and sptSet[v] == False and dist[v] > dist[u] + graph[u][v]:
                dist[v] = dist[u] + graph[u][v]

        dijkstra_sssp(dist)


def floyd_asps(graph):
    num_vertices = len(graph)
    dist = [num_vertices * [0] for i in range(num_vertices)]
    pred = [num_vertices * [0] for i in range(num_vertices)]

    # init loop
    for i in range(len(graph)):
        for j in range(len(graph)):
            dist[i][j] = graph[i][j]  # path of length 1, i.e. just the edge
            pred[i][j] = i  # predecessor will be vertex i
            if dist[i][j] == 0:
                dist[i][j] = sys.maxsize
        dist[i][i] = maxsize  # no cost
        pred[i][i] = -1  # indicates end of path
    # main loop
    for k in range(num_vertices):
        for i in range(num_vertices):
            for j in range(num_vertices):
                if dist[i][j] > dist[i][k] + dist[k][j]:  # use intermediate vertex k
                    dist[i][j] = dist[i][k] + dist[k][j]
                pred[i][j] = pred[k][j]



def print_floyd(dist, graph):
    start = time.time()
    num_vertices = len(graph)
    for i in range(num_vertices):
        for j in range(num_vertices):
            print(dist[i][j])
    end = time.time()
    print(f"florRuntime of the program is {(end - start) * 10000}")
    return floyd_asps(graph)


def main():
    mypath = 'C:\\Users\\Fariha\\PycharmProjects\\pythonProject3\\'
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for file in files:
        if file[0:5] == 'graph' and file.find('_report') < 0:
            analyze_graph(file)


if __name__ == '__main__':
    main()
