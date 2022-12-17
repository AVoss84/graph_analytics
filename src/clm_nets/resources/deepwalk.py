import networkx as nx
import random
import numpy as np
from typing import List
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec

#https://towardsdatascience.com/exploring-graph-embeddings-deepwalk-and-node2vec-ee12c4c0d26d

class DeepWalk:
    def __init__(self, window_size: int, embedding_size: int, walk_length: int, walks_per_node: int):
        """
        :param window_size: window size for the Word2Vec model
        :param embedding_size: size of the final embedding
        :param walk_length: length of the walk
        :param walks_per_node: number of walks per node
        """
        self.window_size = window_size
        self.embedding_size = embedding_size
        self.walk_length = walk_length
        self.walk_per_node = walks_per_node

    def random_walk(self, g: nx.Graph, start: str, use_probabilities: bool = False) -> List[str]:
        """
        Generate a random walk trajectory starting on start
        :param g: Graph
        :param start: starting node for the random walk
        :param use_probabilities: if True take into account the weights assigned to each edge to select the next candidate
        :return:
        """
        walk = [start]
        for i in range(self.walk_length):
            neighbours = g.neighbors(walk[i])
            neighs = list(neighbours)
            if use_probabilities:
                probabilities = [g.get_edge_data(walk[i], neig)["weight"] for neig in neighs]
                sum_probabilities = sum(probabilities)
                probabilities = list(map(lambda t: t / sum_probabilities, probabilities))
                p = np.random.choice(neighs, p=probabilities)
            else:
                p = random.choice(neighs)
            walk.append(p)
        return walk

    def get_walks(self, g: nx.Graph, use_probabilities: bool = False) -> List[List[str]]:
        """
        Generate all the random walks
        :param g: Graph
        :param use_probabilities:
        :return:
        """
        random_walks = []
        for _ in tqdm(range(self.walk_per_node)):
            random_nodes = list(g.nodes)
            random.shuffle(random_nodes)
            for node in random_nodes:
                random_walks.append(self.random_walk(g=g, start=node, use_probabilities=use_probabilities))
        return random_walks

    def compute_embeddings(self, walks: List[List[str]]):
        """
        Compute the node embeddings for the generated walks
        :param walks: List of walks
        :return:
        """
        model = Word2Vec(sentences=walks, window=self.window_size, vector_size=self.embedding_size)
        return model.wv


#----------------------------------------------------------------------------

#start = ''
#walk = [start]
use_probabilities = False
window_size = 5
embedding_size = 100
walk_length = 100
walks_per_node = 20

G = nx.karate_club_graph()



start = random.choice(list(G.nodes()))
start

walk = [start]

G.nodes(data=True)

g = G
i = 0

g.get_edge_data(26,33)

walk[i]
neighbours = g.neighbors(walk[i])
neighbours

neighs = list(neighbours)
neighs

g[26]
g.adj[26]


for i in range(walk_length):
    neighbours = g.neighbors(walk[i])
    neighs = list(neighbours)
    if use_probabilities:
        probabilities = [g.get_edge_data(walk[i], neig)["weight"] for neig in neighs]
        sum_probabilities = sum(probabilities)
        probabilities = list(map(lambda t: t / sum_probabilities, probabilities))
        p = np.random.choice(neighs, p=probabilities)
    else:
        p = random.choice(neighs)
    walk.append(p)


G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc

G.nodes()

[n for n in G.neighbors(0)]


#--------------------------------------------
import networkx as nx

# load graph from networkx library
G = nx.karate_club_graph()

G.edges()

dw = DeepWalk(window_size = 5, embedding_size = 100, walk_length = 10, walks_per_node = 2)

for _ in tqdm(range(dw.walk_per_node)):
    random_nodes = list(g.nodes)
    random.shuffle(random_nodes)
    random_walks = [dw.random_walk(g=g, start=node, use_probabilities=use_probabilities) for node in random_nodes]

random_walks
len(random_walks)

len(random_nodes)

dw.random_walk(g = G, use_probabilities = False)

my_rws = dw.get_walks(g = G, use_probabilities = False)

embed = dw.compute_embeddings(walks = my_rws)

embed.vector_size