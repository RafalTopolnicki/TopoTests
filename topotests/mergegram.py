from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
from scipy.spatial.distance import squareform
import networkx as nx
import numpy as np


def mergegram(data: np.array) -> np.array:
    """
    Returns mergegram introduced by Elkin and Kurlin in https://arxiv.org/abs/2007.11278

    :param data:
    :return:
    """
    dm = distance_matrix(x=data, y=data, p=2)
    dists = squareform(dm)
    hls = hierarchy.linkage(dists, "single")
    # get mergerogram form linkage matrix
    npts = data.shape[0]
    last_cluster_id = npts - 1
    born_clusters = {}
    for i in range(npts):
        # FIXME: is float index for dict makes sense?
        born_clusters[float(i)] = 0
    pts = []
    for hl in hls:
        last_cluster_id += 1
        x1, x2, t = hl[0], hl[1], hl[2]
        y1 = born_clusters[x1]
        y2 = born_clusters[x2]
        pts.append([y1 / 2, t / 2])
        pts.append([y2 / 2, t / 2])
        born_clusters[last_cluster_id] = t
    # change output to an array
    pts = np.array(pts)
    return pts

def pathgram(data: np.array) -> np.array:
    """

    :param data:
    :return:
    """
    data = np.array([[0, 0], [1, 0], [100, 0], [150, 0]])
    dm = distance_matrix(x=data, y=data, p=2)
    dists = squareform(dm)
    hls = hierarchy.linkage(dists, "single")
    dendro = hierarchy.dendrogram(hls)

    # # Create an empty NetworkX graph
    # G = nx.Graph()
    # # Iterate over the dendrogram dictionary to extract information
    # for i, d in zip(dendro['icoord'], dendro['dcoord']):
    #     for x, y in zip(i, d):
    #         # Add nodes to the graph
    #         if x not in G:
    #             G.add_node(x)
    #         if y not in G:
    #             G.add_node(y)
    #         # Add edges to the graph
    #         G.add_edge(x, y)
    # # Now you have converted the dendrogram to a NetworkX graph
    print("done")
    return pts