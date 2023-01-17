import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform



POPULATION = 10000
NUM_CITIES = 100


locs = np.random.rand(NUM_CITIES, 2)*100
population = np.random.randint(1, 3, NUM_CITIES).astype('int64')

while sum(population)<POPULATION+1:
    if sum(population)<POPULATION*0.3:
        population += population * np.random.randint(0, 5, NUM_CITIES).astype('int64')
    else:
        if sum(population)>POPULATION-1:
            break
        population += np.eye(NUM_CITIES)[np.random.choice(NUM_CITIES, 1)].astype('int64').reshape(100,) 

A = np.outer(population,population)/(np.sqrt(squareform(pdist(locs)))+np.eye(len(locs)))
np.fill_diagonal(A, 0)
A = A/np.max(A)

G = nx.from_numpy_array(A)
node_colors = [0] * len(A)
node_sizes = population.tolist()
cmap = plt.cm.jet

pos = {
        n: locs[i]
        for i, n in enumerate(G.nodes)
    }


for (u,v,d) in list(G.edges(data=True)):
    if d["weight"] <= 0.12:
        G.remove_edge(u, v)

plt.figure(figsize=(10,10))
nx.draw_networkx(G,
                pos,
                with_labels=None,
                node_color=node_colors,
                node_size=node_sizes,
                edge_color='c',
                alpha=0.6, 
                cmap=cmap)

plt.grid(True)
plt.show()