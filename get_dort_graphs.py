import numpy as np
import networkx as nx
import os.path as path

def get_dort_graphs(ds_name):
    "WORK IN PROGRESS"
    graphs = {}
    data_dir = "./data/"
    edges = np.array([(x[0], x[1]) for x in np.loadtxt(data_dir + ds_name + '/' + ds_name + "_A.txt",
                     delimiter = ',', dtype = 'int')])
    ind = np.loadtxt(data_dir + ds_name + '/' + ds_name + "_graph_indicator.txt", dtype = 'int')
    _, indices = np.unique(ind, return_index = True)
    edges_split = np.split(edges, indices)
    for i,edges in enumerate(edges_split):
        edges_tuples = list(map(tuple, edges))
        graphs[i] = nx.Graph()
        graphs[i].add_edges_from(edges_tuples)

    return graphs

ds_name = "PTC_MR"
K = get_dort_graphs(ds_name)
