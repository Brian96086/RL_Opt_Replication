import numpy as np
from numpy import random
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import gym
import copy
import datetime

from .EpiModel import EpiModel

#bound for (population, symp_city/pop_city, symp_all/pop_all, recovered_city/pop_city, 
    # dead_city/pop_city, ExpPopIn_city, local_inc_city/pop_city, local_inc_all/pop_all)
low_bound = np.array([0, 0, 0, 0, 0, 0, -1, -1])
up_bound = np.array([10000, 1, 1, 1, 1, 10000, 1, 1])

class Game(MultiAgentEnv):
    
    def __init__(self):
        super(Game, self).__init__()
        self.NUM_CITIES = 100
        
        self.NUM_INITIAL_INFECTED = 50
        self.POPULATION = 10000
        
        self.num_agents = 100
        self._agent_ids = list(range(self.num_agents))
        
        self.action_space_dict = dict(zip(self._agent_ids, [gym.spaces.Discrete(2)] * self.num_agents))
        self.observation_space = dict(zip(self._agent_ids, [
            gym.spaces.Box(low=low_bound, high=up_bound, shape=(8,))] * self.num_agents))

        self.C_DEAD = 5
        self.C_INF = 1
#         self.C_STEPLOCK = 2*(10**2)
        self.C_LOCK = 1/364
        self.C_ALPHA = 2
#         self.C_BETA = 0.001

        self.week = 0
        self.max_week = 52

        self.global_susceptible = np.zeros(364)
        self.global_exposed = np.zeros(364)
        self.global_asymptomatic = np.zeros(364)
        self.global_symptomatic = np.zeros(364)
        self.global_recovered = np.zeros(364)
        self.global_dead = np.zeros(364)
        self.global_lockdown = np.zeros(364)

        self.susceptible = np.zeros(self.NUM_CITIES)
        self.exposed = np.zeros(self.NUM_CITIES)
        self.asymptomatic = np.zeros(self.NUM_CITIES)
        self.symptomatic = np.zeros(self.NUM_CITIES)
        self.recovered = np.zeros(self.NUM_CITIES)
        self.dead = np.zeros(self.NUM_CITIES)
        self.lockdown = np.zeros(self.NUM_CITIES)

        self.u_onoff = np.ones([self.NUM_CITIES,364])
        self.local_increase = np.zeros(self.NUM_CITIES)
        self.death_increase = np.zeros(self.NUM_CITIES)

        self.A, self.population = self.make_world()
        self.initial_infected = self.make_initial_infected()

        self.ExpPopIn = np.zeros(self.NUM_CITIES)
        self.episode_count = 0
        
    
    def step(self, action_dict):
        
        if self.week > 0:
            for city in range(self.NUM_CITIES):
                self.u_onoff[city,7*self.week:7*self.week+8] = action_dict[city]

        symptomatic_prev = copy.deepcopy(self.symptomatic)
        dead_prev = copy.deepcopy(self.dead)

        self.susceptible,self.exposed,self.asymptomatic,self.symptomatic,self.recovered,self.dead,\
        self.global_susceptible,self.global_exposed,self.global_asymptomatic,self.global_symptomatic,\
        self.global_recovered,self.global_dead,self.ExpPopIn =\
        self.run_simulation()
        
        self.local_increase = self.symptomatic - symptomatic_prev
        self.death_increase = self.dead - dead_prev
        
        reward = {city: self.get_reward(city) for city in range(100)}
        
        observation = {city: self.get_observation(city) for city in range(100)}
        
        self.week += 1
        if self.week == self.max_week:
            
            self.episode_count += 1
            
            if self.episode_count % 20 == 0:
            
                colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

                fig, ax = plt.subplots(1)
                (pd.DataFrame(self.global_susceptible)).plot(color=colors[0], linestyle='-', label='Susceptible', ax=ax)
                (pd.DataFrame(self.global_exposed)).plot(color=colors[4], linestyle='-', label='Exposed', ax=ax)
                (pd.DataFrame(self.global_asymptomatic + self.global_asymptomatic)).plot(color=colors[1], linestyle=':', label='Infected', ax=ax)
                (pd.DataFrame(self.global_lockdown)).plot(color=colors[5], linestyle='-', label='In Lockdown', ax=ax)
                (pd.DataFrame(self.global_recovered)).plot(color=colors[2], linestyle='-', label='Recovered', ax=ax)
                (pd.DataFrame(self.global_dead)).plot(color=colors[7], linestyle='-', label='Dead', ax=ax)


                # ax.legend(['Susceptible','Asymptomatic','Symptomatic','Recovered','Dead'])
                ax.legend(['Susceptible','Exposed','Infected','In Lockdown','Recovered','Dead'])
                ax.set_xlabel('Time')
                ax.set_ylabel('Population')
    #             filename  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = "policy_" + str(self.episode_count/20)
                plt.savefig("results/policy/{0}.png".format(filename))
                plt.close()
            
        done = {"__all__": self.week >= self.max_week,}
        info = {}
        
        return observation, reward, done, info


    def get_reward(self,city):
        return (self.get_step_reward(city)+self.get_terminal_reward(city))/2


    def get_step_reward(self, city):
#         print("city: ", city)
#         print("death increase: ", self.death_increase[city])
#         print("local infection increase: ", self.local_increase[city])
#         print("on off: ", self.u_onoff[city, 7*(self.week-1)])
#         print("reward: ", (self.C_DEAD*self.death_increase[city] + self.C_INF*self.local_increase[city] + self.C_STEPLOCK*(1-self.u_onoff[city, 7*(self.week-1)]))/self.population[city])
#         print("--------------")
        return -1 * (self.C_DEAD*self.death_increase[city] + self.C_INF*self.local_increase[city] + self.C_LOCK*(1-self.u_onoff[city, 7*(self.week-1)]))/self.population[city]


    def get_terminal_reward(self, city):
        return self.C_ALPHA-((self.C_DEAD*self.dead[city]/self.population[city]) + self.C_INF*(self.symptomatic[city]+self.symptomatic[city])/self.population[city] + self.C_LOCK*(364-sum(self.u_onoff[city])))


    #dimension = 8 -> (population, symp_city/pop_city, symp_all/pop_all, recovered_city/pop_city, 
    # dead_city/pop_city, ExpPopIn_city, local_inc_city/pop_city, local_inc_all/pop_all)
    def get_observation(self, city):
        return np.hstack([self.population[city], self.symptomatic[city]/self.population[city], sum(self.symptomatic)/sum(self.population),\
                           self.recovered[city]/self.population[city], self.dead[city]/self.population[city], self.ExpPopIn[city],\
                          self.local_increase[city]/self.population[city], sum(self.local_increase)/sum(self.population)])

    def reset(self):
        self.week = 0
        self.global_susceptible = np.zeros(364)
        self.global_exposed = np.zeros(364)
        self.global_asymptomatic = np.zeros(364)
        self.global_symptomatic = np.zeros(364)
        self.global_recovered = np.zeros(364)
        self.global_dead = np.zeros(364)

        self.susceptible = np.zeros(self.NUM_CITIES)
        self.exposed = np.zeros(self.NUM_CITIES)
        self.asymptomatic = np.zeros(self.NUM_CITIES)
        self.symptomatic = np.zeros(self.NUM_CITIES)
        self.recovered = np.zeros(self.NUM_CITIES)
        self.dead = np.zeros(self.NUM_CITIES)

        self.u_onoff = np.ones([self.NUM_CITIES,364])

        self.local_increase = np.zeros(self.NUM_CITIES)
        self.death_increase = np.zeros(self.NUM_CITIES)

        return {city: self.get_observation(city) for city in range(100)}

    
    def render(self, mode):
        pass
    
    
    def close(self):
        pass

    
    def seed(self):
        pass


    def run_simulation(self):
        #if(self.week%1)
        print("simulation starts at week = ", self.week)
        week = self.week
        u_onoff = self.u_onoff

        ExpPopIn_a = np.zeros(self.NUM_CITIES)
        for day in range(7):
            for city in range(self.NUM_CITIES):

                if week == 0 and day == 0:
                    I0 = self.initial_infected[city]
                    N = self.population[city]
                    S = N-I0
                    Ia = 0
                    Is = I0
                    E = 0
                    R = 0
                    D = 0
                else:
                    S = self.susceptible[city]
                    Ia = self.asymptomatic[city]
                    Is = self.symptomatic[city]
                    E = self.exposed[city]
                    R = self.recovered[city]
                    D = self.dead[city]

                SEIIRD = EpiModel()

                rbeta = 0.95
                pa = 0.2
                R0 = 1.7

                beta = .9
                mu = 0.04

                epsilon = .07

                PD = .02
                #print("curr_day {0} = , city = {1}, u_onoff = {2}".format(7*week+day, city, self.u_onoff[city,7*week+day]))
                if self.u_onoff[city,7*week+day] == 1:
                    SEIIRD.add_spontaneous('S', 'E', .07)
                    SEIIRD.add_interaction('S', 'E', 'Is',  0.1)
                    SEIIRD.add_interaction('S', 'E', 'Ia',  0.1)
                    ExpPopIn = 0
                    for othercity in range(self.NUM_CITIES):
                        ExpPopIn += self.A[city,othercity]*self.u_onoff[othercity,day]*self.asymptomatic[othercity]
                    SEIIRD.add_interaction('S', 'E', 'ExpPopIn', 0.1)
                    SEIIRD.add_spontaneous('E', 'Ia', epsilon*pa)
                    SEIIRD.add_spontaneous('E', 'Is', epsilon*(1-pa))
                    SEIIRD.add_spontaneous('Ia', 'R', mu)
                    SEIIRD.add_spontaneous('Is', 'R', (1-PD)*mu)
                    SEIIRD.add_spontaneous('Is', 'D', PD*mu)
                    SEIIRD.integrate(3, S=S, Ia=Ia, Is=Is, E=E, R=R, D=D, ExpPopIn=ExpPopIn)
                    self.lockdown[city] = 0
                else:
                    #print("ExpPopIn = ", ExpPopIn)
                    #print("else")
                    ExpPopIn = 0
                    SEIIRD.add_spontaneous('S', 'E', 0)
                    SEIIRD.add_interaction('S', 'E', 'Ia', 0.01)
                    SEIIRD.add_spontaneous('E', 'Ia', epsilon*pa)
                    SEIIRD.add_spontaneous('E', 'Is', epsilon*(1-pa))
                    SEIIRD.add_spontaneous('Ia', 'R', mu)
                    SEIIRD.add_spontaneous('Is', 'R', (1-PD)*mu)
                    SEIIRD.add_spontaneous('Is', 'D', PD*mu)
                    SEIIRD.integrate(3, S=S, Ia=Ia, Is=Is, E=E, R=R, D=D)
                    self.lockdown[city] = self.population[city]

                self.susceptible[city] = SEIIRD.S[2]
                self.exposed[city] = SEIIRD.E[2]
                self.asymptomatic[city] = SEIIRD.Ia[2]
                self.symptomatic[city] = SEIIRD.Is[2]
                self.recovered[city] = SEIIRD.R[2]
                self.dead[city] = SEIIRD.D[2]
                #print("day = ", 7*week+day)
                if(self.susceptible[city]<0):print("suspectible<0")
                if(self.exposed[city]<0):print("exposed<0")
                if(self.asymptomatic[city]<0):print("asymptomatic<0")
                if(self.recovered[city]<0):print("recovered<0")
                if(self.dead[city]<0):print("dead<0")
                if(self.ExpPopIn[city]<0):print("ExpPopIn<0")

                ExpPopIn_a[city] = ExpPopIn

            self.global_susceptible[7*week+day] = sum(self.susceptible)
            self.global_exposed[7*week+day] = sum(self.exposed)
            self.global_asymptomatic[7*week+day] = sum(self.asymptomatic)
            self.global_symptomatic[7*week+day] = sum(self.symptomatic)
            self.global_recovered[7*week+day] = sum(self.recovered)
            self.global_dead[7*week+day] = sum(self.dead)
            self.global_lockdown[7*week+day] = sum(self.lockdown)
        #print("simulation done at week = ", self.week)
        return self.susceptible,self.exposed,self.asymptomatic,self.symptomatic,self.recovered,self.dead,\
          self.global_susceptible,self.global_exposed,self.global_asymptomatic,self.global_symptomatic,self.global_recovered,self.global_dead, ExpPopIn_a

    
    def make_world(self):
        
        locs = np.random.rand(self.NUM_CITIES, 2)*100
        population = np.random.randint(1, 3, self.NUM_CITIES).astype('int64')

        while sum(population)<self.POPULATION+1:
            if sum(population)<self.POPULATION*0.3:
                #for each city, multiply the population by random factor within[1,6) (or add by [0,5))
                population += population * np.random.randint(0, 5, self.NUM_CITIES).astype('int64')
            else:
                if sum(population)>self.POPULATION-1:
                    break
                #randomly choose a city(in row vector) and adds 1 person to that city
                population += np.eye(self.NUM_CITIES)[np.random.choice(self.NUM_CITIES, 1)].astype('int64').reshape(100,) 

        A = np.outer(population,population)/(np.sqrt(squareform(pdist(locs)))+np.eye(len(locs)))
        np.fill_diagonal(A, 0)
        A = A/np.max(A)
        
        return A, population


    def make_initial_infected(self):
        total = self.NUM_INITIAL_INFECTED
        initial_infected = []
        for i in range(self.NUM_CITIES-1):
            val = np.random.randint(0, total)
            initial_infected.append(val)
            total -= val
        initial_infected.append(total)
        
        return initial_infected
        