import random
import numpy as np
import matplotlib.pyplot as plt
import math


DIMENSION = 30
POP_SIZE = 100
MUTATION_RATE = 0.1
LEFT_BOUNDARY = -500
RIGHT_BOUNDARY = 500
GENERATION = 200


class GeneticAlgorithm(object):
    def __init__(self)->None:
        pass

    def crossover(self, chromosome1, chromosome2):
        point = int(random.uniform(1, DIMENSION-1) + 0.5)
        chromosome1_front = chromosome1[:point]
        chromosome1_back = chromosome1[point:]
        chromosome2_front = chromosome2[:point]
        chromosome2_back = chromosome2[point:]
        chromosome3 = np.append(chromosome1_front, chromosome2_back)
        chromosome4 = np.append(chromosome2_front, chromosome1_back)
        return chromosome3, chromosome4
    
    def mutate(self, chromosome):
        point = int(random.uniform(0, DIMENSION-1) + 0.5)
        chromosome[point] = random.uniform(LEFT_BOUNDARY, RIGHT_BOUNDARY)
    
    def evaluateBySphereFunction(self, chromosome):
        sum = 0
        for num in chromosome:
            sum += num * num
        return sum
    
    def initchromosome(self):
        chromosome = np.empty(DIMENSION, dtype=float)
        for i in range(DIMENSION):
            chromosome[i] = random.uniform(-100,100)
        return chromosome
    
    def initPopulation(self):
        pop = []
        for i in range(int(POP_SIZE)):
            c = self.initchromosome()
            pop.append(c)
        # for i in range(int(POP_SIZE/2)):
        #     num = random.uniform(-100,100)
        #     c = np.full(DIMENSION, num)
        #     pop.append(c)
        return pop
    
    def generateOffSpring(self, chromosome1, chromosome2):
        c1, c2 = self.crossover(chromosome1=chromosome1, chromosome2=chromosome2)
        if(random.random()<=MUTATION_RATE):
            self.mutate(c1)
        if(random.random()<=MUTATION_RATE):
            self.mutate(c2)
        return np.copy(c1), np.copy(c2)
    
    def getChildPopulation(self, fa):
        child = []
        while len(child) < POP_SIZE:
            f1 = random.choice(fa)
            f2 = random.choice(fa)
            c1, c2 = self.generateOffSpring(f1,f2)
            child.append(c1)
            child.append(c2)
        return child

    def evaluateByF8(self, chromosome):
        c2 = np.reshape(chromosome,[DIMENSION,1])
        sum = -np.dot(np.sin(np.sqrt(np.abs(chromosome))), c2)
        return sum









y = np.empty(0, dtype=float)
algorithm = GeneticAlgorithm()
fa = algorithm.initPopulation()
y = np.append(y, algorithm.evaluateByF8(fa[0]))
for i in range(GENERATION):
    children = algorithm.getChildPopulation(fa=fa)
    total = fa + children
    total.sort(key=algorithm.evaluateByF8)
    fa = total[:POP_SIZE]
    y = np.append(y, algorithm.evaluateByF8(fa[0]))
print(fa[0])
print(algorithm.evaluateByF8(fa[0]))

x = np.arange(GENERATION + 1)
plt.plot(x,y)
plt.show()



        
        


