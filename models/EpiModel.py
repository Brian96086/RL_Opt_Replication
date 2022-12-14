import numpy as np
from numpy import random
import pandas as pd
import networkx as nx
import scipy
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

class EpiModel(object):
# Taken from https://github.com/DataForScience/Epidemiology101/
    def __init__(self, compartments=None):
        self.transitions = nx.MultiDiGraph()
        
        if compartments is not None:
            self.transitions.add_nodes_from([comp for comp in compartments])
    
    def add_interaction(self, source, target, agent, rate):        
        self.transitions.add_edge(source, target, agent=agent, rate=rate)        
        
    def add_spontaneous(self, source, target, rate):
        self.transitions.add_edge(source, target, rate=rate)
        
    def _new_cases(self, population, time, pos):
        diff = np.zeros(len(pos))
        N = np.sum(population)        
        
        for edge in self.transitions.edges(data=True):
            source = edge[0]
            target = edge[1]
            trans = edge[2]
            
            rate = trans['rate']*population[pos[source]]
            
            if 'agent' in trans:
                agent = trans['agent']
                rate *= population[pos[agent]]/N
                
            diff[pos[source]] -= rate
            diff[pos[target]] += rate
            
        return diff
    
    def __getattr__(self, name):
        if 'values_' in self.__dict__:
            return self.values_[name]
        else:
            raise AttributeError("'EpiModel' object has no attribute '%s'" % name)

    def simulate(self, timesteps, **kwargs):
        pos = {comp: i for i, comp in enumerate(kwargs)}
        population=np.zeros(len(pos), dtype='int')

        for comp in pos:
            population[pos[comp]] = kwargs[comp]

        values = []
        values.append(population)

        comps = list(self.transitions.nodes)
        time = np.arange(1, timesteps, 1, dtype='int')

        for t in time:
            pop = values[-1]
            new_pop = values[-1].copy()
            N = np.sum(pop)


            for comp in comps:
                trans = list(self.transitions.edges(comp, data=True))             

                prob = np.zeros(len(comps), dtype='float')

                for _, node_j, data in trans:
                    source = pos[comp]
                    target = pos[node_j]

                    rate = data['rate']

                    if 'agent' in data:
                        agent = pos[data['agent']]
                        rate *= pop[agent]/N

                    prob[target] = rate

                prob[source] = 1-np.sum(prob)

                delta = random.multinomial(pop[source], prob)
                delta[source] = 0

                changes = np.sum(delta)

                if changes == 0:
                    continue

                new_pop[source] -= changes

                for i in range(len(delta)):
                    new_pop[i] += delta[i]

            values.append(new_pop)

        values = np.array(values)
        self.values_ = pd.DataFrame(values[1:], columns=comps, index=time)
    
    def integrate(self, timesteps, **kwargs):
        pos = {comp: i for i, comp in enumerate(kwargs)}
        population=np.zeros(len(pos))
        
        for comp in pos:
            population[pos[comp]] = kwargs[comp]
        
        time = np.arange(1, timesteps, 1)

        self.values_ = pd.DataFrame(scipy.integrate.odeint(self._new_cases, population, time, args=(pos,)), columns=pos.keys(), index=time)