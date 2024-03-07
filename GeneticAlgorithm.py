# =================================
# Editor: Jingyuan Xu
# Create Date: 2024/3/4
# Description: The implementation class of genetic algorithm encapsulates the operators related to genetic algorithm
# =================================
from dependency import *

POP_SIZE = 100
MUTATION_RATE = 0.1
GENERATION = 400

class GeneticAlgorithm(object):
    def __init__(self, left, right, dimension):
        self.DIMENSION = dimension
        self.POP_SIZE = POP_SIZE
        self.MUTATE_RATE = MUTATION_RATE
        self.LEFT_BOUNDARY = left
        self.RIGHT_BOUNDARY = right
        self.GENERATION = GENERATION
        pass

    def crossover(self, chromosome1, chromosome2):
        point = int(random.uniform(1, self.DIMENSION-1) + 0.5)
        chromosome1_front = chromosome1[:point]
        chromosome1_back = chromosome1[point:]
        chromosome2_front = chromosome2[:point]
        chromosome2_back = chromosome2[point:]
        chromosome3 = np.append(chromosome1_front, chromosome2_back)
        chromosome4 = np.append(chromosome2_front, chromosome1_back)
        return chromosome3, chromosome4
    
    def mutate(self, chromosome):
        point = int(random.uniform(0, self.DIMENSION-1) + 0.5)
        chromosome[point] = random.uniform(self.LEFT_BOUNDARY, self.RIGHT_BOUNDARY)
    
    def initchromosome(self):
        chromosome = np.empty(self.DIMENSION, dtype=float)
        for i in range(self.DIMENSION):
            chromosome[i] = random.uniform(self.LEFT_BOUNDARY,self.RIGHT_BOUNDARY)
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