"""
moead.py

Description:

A Python implementation of the decomposition based multi-objective evolutionary algorithm (MOEA/D).

MOEA/D is described in the following publication: Zhang, Q. & Li, H. MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition. IEEE Trans. Evol. Comput. 11, 712-731 (2007).

The code in moead.py is a port of the MOEA/D algorithm provided by jmetal.metaheuristics.moead.MOEAD.java from the JMetal Java multi-objective metaheuristic algorithm framework (http://jmetal.sourceforge.net/). The JMetal specific functions have been converted to the DEAP (Distributed Evolutionary Algorithms in Python, http://deap.readthedocs.io/en/master/) equivalence.

The current version works for the knapsack example and appears to show adequate exploration of the pareto front solutions. It would be preferable to test additional problems to determine whether this works as intended different MOEA requirements (problems different combinations of maximization and minimization objective functions, for example.)

Note also that weight vectors are only computed for populations of size 2 or 3. Problems with 4 or more objectives will requires a weights file in the "weights" directory. Weights can be downloaded from: http://dces.essex.ac.uk/staff/qzhang/MOEAcompetition/CEC09final/code/ZhangMOEADcode/moead030510.rar

Author: Manuel Belmadani <mbelm006@uottawa.ca>
Date  : 2016-09-05

"""
from exceptions import NotImplementedError
from deap import tools

import random
import math
from copy import deepcopy

class MOEAD(object):

    def __init__(self, population, toolbox, mu, cxpb, mutpb, ngen=0, maxEvaluations=0,
                 T=20, nr=2, delta=0.9, stats=None, halloffame=None, verbose=__debug__, dataDirectory="weights"):

        self.populationSize_ = int(0)

        # Reference the DEAP toolbox to the algorithm 
        self.toolbox = toolbox

        # Stores the population
        self.population = []

        if population:
            self.population = population
            fitnesses = self.toolbox.map(self.toolbox.evaluate, self.population)
            for ind, fit in zip(self.population, fitnesses):
                ind.fitness.values = fit
            
            self.populationSize_ = mu

        # Reference the DEAP toolbox to the algorithm 
        self.toolbox = toolbox

        # Z vector (ideal point)
        self.z_ = [] # of type floats

        # Lambda vectors (Weight vectors)
        self.lambda_ = [] # of type list of floats, i.e. [][], e.g. Vector of vectors of floats

        # Neighbourhood size
        self.T_ = int(0)

        # Neighbourhood
        self.neighbourhood_ = [] # of type int, i.e. [][], e.g. Vector of vectors of integers

        # delta: the probability that parent solutions are selected from neighbourhoods
        self.delta_ = delta

        # nr: Maximal number of individuals replaced by each child
        self.nr_ = nr
        
        self.indArray_ = list() # of type Individual
        self.functionType_ = str()
        self.evaluations_ = int()

        # Operators
        ## By convention, toolbox.mutate and toolbox.mate should handle crossovers and mutations.
        try:
            self.toolbox.mate
            self.toolbox.mutate
        except Exception as e:
            print "Error in MOEAD.__init__: toolbox.mutate or toolbox.mate is not assigned."
            raise e                     

        ### Additional stuff not from JMetal implementation
        
        self.n_objectives = len(self.population[0].fitness.values)
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.ngen = ngen
        self.paretoFront = halloffame
        
        self.maxEvaluations = -1
        if maxEvaluations == ngen == 0:
            print "maxEvaluations or ngen must be greater than 0."
            raise ValueError
        if ngen > 0:
            self.maxEvaluations =  ngen * self.populationSize_
        else:
            self.maxEvaluations = maxEvaluations
        
        self.dataDirectory_ = dataDirectory
        self.functionType_ = "_TCHE1"
        self.stats = stats
        self.verbose = verbose

        ### Code brough up from the execute function
        self.T_ = T
        self.delta_ = delta
        
    def execute(self):
        print "Executing MOEA/D"

        logbook = tools.Logbook()
        logbook.header = ['gen', 'evals'] + (self.stats.fields if self.stats else [])
        
        self.evaluations_ = 0
        print "POPSIZE:", self.populationSize_
    
        self.indArray_ = [ self.toolbox.individual() for _ in range(self.n_objectives) ]

        # 2-D list of size populationSize * T_
        self.neighbourhood_ = [[None] * self.T_] * (self.populationSize_)  

        # List of size number of objectives. Contains best fitness.
        self.z_ = self.n_objectives * [None]
        
        #2-D list  Of size populationSize_ x Number of objectives. Used for weight vectors.
        self.lambda_ = [[None] * self.n_objectives] * self.populationSize_ 

        # STEP 1. Initialization
        self.initUniformWeight()
        self.initNeighbourhood()
        self.initIdealPoint()
        
        record = self.stats.compile(self.population) if self.stats is not None else {}
        
        logbook.record(gen=self.ngen , evals=self.evaluations_, **record)
        if self.verbose: print logbook.stream
        
        while self.evaluations_ < self.maxEvaluations:
            permutation = [None] * self.populationSize_ # Of type int
            self.randomPermutations(permutation, self.populationSize_)

            for i in xrange(self.populationSize_):
                n = permutation[i]
                type_ = int()
                rnd = random.random()

                # STEP 2.1: Mating selection based on probability
                if rnd < self.delta_:
                    type_ = 1
                else:
                    type_ = 2

                p = list() # Vector of type integer
                self.matingSelection(p, n, 2, type_)

                # STEP 2.2: Reproduction
                child = None
                children = []
                parents = [None] * 3


                candidates = list(self.population[:])
                parents[0] = deepcopy(candidates[p[0]])
                parents[1] = deepcopy(candidates[p[1]])

                children = self.toolbox.mate(parents[0], parents[1])                
                
                # Apply mutation
                children = [self.toolbox.mutate(child) for child in children]
                # Evaluation
                offspring = []

                for child in children:
                    fit = self.toolbox.evaluate(child[0])
                    self.evaluations_ += 1
                    child[0].fitness.values = fit
                    offspring.append(child[0])


                # STEP 2.3 Repair
                # TODO: Add this as an option to repair invalid individuals?

                # STEP 2.4: Update z_
                for child in offspring:
                    self.updateReference(child)

                    # STEP 2.5 Update of solutions
                    self.updateProblem(child, n, type_)
                            
                record = self.stats.compile(self.population) if self.stats is not None else {}

                logbook.record(gen=self.ngen, evals=self.evaluations_, **record )

                if self.verbose: print logbook.stream

                
        return self.population

    """
    " initUniformWeight
    """
    def initUniformWeight(self):
        """
        Precomputed weights from (Zhang, Multiobjective Optimization Problems With Complicated Pareto Sets, MOEA/D and NSGA-II) downloaded from: 
        http://dces.essex.ac.uk/staff/qzhang/MOEAcompetition/CEC09final/code/ZhangMOEADcode/moead030510.rar)
        
        """
        if self.n_objectives == 2: 
            for n in xrange(self.populationSize_):
                a = 1.0 * float(n) / (self.populationSize_ - 1)
                self.lambda_[n][0] = a
                self.lambda_[n][1] = 1 - a
        elif self.n_objectives == 3:
            """
            Ported from Java code written by Wudong Liu 
            (Source: http://dces.essex.ac.uk/staff/qzhang/moead/moead-java-source.zip)
            """
            m = self.populationSize_

            self.lambda_ = list()
            for i in xrange(m):
                for j in xrange(m):
                    if i+j <= m:
                        k = m - i - j
                        try:
                            weight_scalars = [None] * 3
                            weight_scalars[0] = float(i) / (m)
                            weight_scalars[1] = float(j) / (m)
                            weight_scalars[2] = float(k) / (m)
                            self.lambda_.append(weight_scalars)
                        except Exception as e:
                            print "Error creating weight with 3 objectives at:"
                            print "count", count
                            print "i", i
                            print "j", j
                            print "k", k
                            raise e
            # Trim number of weights to fit population size
            self.lambda_ = sorted((x for x in self.lambda_), key=lambda x: sum(x), reverse=True)
            self.lambda_ = self.lambda_[:self.populationSize_]
        else:
            dataFileName = "W" + str(self.n_objectives) + "D_" + str(self.populationSize_) + ".dat"
            file_ = self.dataDirectory_ + "/" + dataFileName
            try:
                with open(file_, 'r') as f:
                    numberOfObjectives = 0
                    i = 0
                    for aux_ in f:
                        j = 0
                        aux = aux_.strip()
                        tokens = aux.split(" ")
                        n_objectives = len(tokens)
                        try:
                            for value in tokens:
                                self.lambda_[i][j] = float(value)
                                j += 1
                        except Exception as e:
                            print "Error loading floats as weight vectors"
                            print "tokens", tokens
                            print "value:",  value
                            print "lambda_", len(self.lambda_),"*",len(self.lambda_[0])
                            print i, j, aux_
                            print e
                            raise e
                        i += 1
                f.close()
            except Exception as e:
                print "initUniformWeight: failed when reading for file:", file_
                print e
                raise e

    """ 
    " initNeighbourhood
    """
    def initNeighbourhood(self):
        x = [None] * self.populationSize_ # Of type float
        idx = [None] * self.populationSize_ #Of type int

        for i in xrange(self.populationSize_):
            for j in xrange(self.populationSize_):
                x[j] = self.distVector(self.lambda_[i], self.lambda_[j])
                idx[j] = j

            self.minFastSort(x, idx, self.populationSize_, self.T_)
            self.neighbourhood_[i][0:self.T_] = idx[0:self.T_] #System.arraycopy(idx, 0, neighbourhood_[i], 0, T_)


    """
    " initPopulation
    " Not implemented: population should be passed as argument
    """
    def initPopulation(self):
        for i in xrange(self._populationSize):
            if True:
                continue
            # generated solutions
            # evaluate solutions
            self.evaluations_ += 1
            # add solution to population           
        raise NotImplementedError

    """
    " initIdealPoint
    """ 
    def initIdealPoint(self):
        for i in range(self.n_objectives):            
            self.indArray_[i] = self.toolbox.individual()
            self.indArray_[i].fitness.values = self.toolbox.evaluate(self.indArray_[i])
            self.z_[i] = 1e30 * self.indArray_[i].fitness.weights[i] # mbelmadani: For minimization objectives
            self.evaluations_+=1
    
        for i in xrange(self.populationSize_): #?
            self.updateReference(self.population[i])

    """
    " matingSelection
    """
    def matingSelection(self, vector, cid, size, type_):
        # vector : the set of indexes of selected mating parents
        # cid    : the id o current subproblem
        # size   : the number of selected mating parents
        # type   : 1 - neighborhood; otherwise - whole population
        """
        Selects 'size' distinct parents, 
        either from the neighbourhood (type=1) or the populaton (type=2).
        """
        
        ss = int()
        r = int()
        p = int()

        ss = len(self.neighbourhood_[cid])
        while len(vector) < size:
            if type_ == 1:
                r = random.randint(0, ss - 1)
                p = self.neighbourhood_[cid][r]
            else:
                p = random.randint(0, self.populationSize_ - 1)
            flag = True
            for i in xrange(len(vector)): 
                if vector[i] == p: # p is in the list
                    flag = False
                    break
            if flag:
                vector.append(p)

    """
    " updateReference
    " @param individual
    """
    def updateReference(self, individual):
        for n in xrange(self.n_objectives):
            if individual.fitness.values[n] < self.z_[n]:
                self.z_[n] = individual.fitness.values[n] * individual.fitness.weights[n] 
                self.indArray_[n] = individual

    """
    " updateProblem
    " @param individual
    " @param id
    " @param type
    """
    def updateProblem(self, individual, id_, type_):
        """
        individual : A new candidate individual
        id : index of the subproblem
        type : update solutions in neighbourhood (type = 1) or whole population otherwise.
        """
        size = int()
        time = int()

        time = 0

        if type_ == 1:
            size = len(self.neighbourhood_[id_])
        else:
            size = len(self.population)
        perm = [None] * size

        self.randomPermutations(perm, size)

        for i in xrange(size):
            k = int()
            if type_ == 1:
                k = self.neighbourhood_[id_][perm[i]]
            else:
                k = perm[i]

            f1, f2 = float(), float()

            f1 = self.fitnessFunction( self.population[k], self.lambda_[k] )
            f2 = self.fitnessFunction( individual, self.lambda_[k] )
            
            #if f2 < f1: # minimization, JMetal default
            if f2 >= f1: # maximization assuming DEAP weights paired with fitness
                self.population[k] = individual
                time += 1
            if time >= self.nr_:
                self.paretoFront.update(self.population)
                return

    """
    " fitnessFunction
    " @param individual
    " @param lambda_
    """
    def fitnessFunction(self, individual, lambda_):
        fitness = float()

        if self.functionType_ == "_TCHE1":
            maxFun = -1.0e+30 
            for n in xrange(self.n_objectives):                
                diff = abs(individual.fitness.values[n] - self.z_[n]) # JMetal default
                feval = float()
                if lambda_[n] == 0:
                    feval = 0.0001 * diff
                else:
                    feval =  diff * lambda_[n]

                if feval > maxFun:
                    maxFun = feval

            fitness = maxFun 
        else:
            print "MOEAD.fitnessFunction: unknown type", self.functionType_
            raise NotImplementedError
        
        return fitness
    
    #######################################################################
    # Ported from the Utils.java class
    #######################################################################
    def distVector(self, vector1, vector2 ):
        dim = len(vector1)
        sum_ = 0
        for n in xrange(dim):
            sum_ += ((vector1[n] - vector2[n] ) * (vector1[n] - vector2[n]))
        return math.sqrt(sum_)

    def minFastSort(self, x, idx, n, m ):
        """
        x   : list of floats
        idx : list of integers (each an index)
        n   : integer
        m   : integer        
        """
        for i in xrange(m):
            for j in xrange(i+1, n):
                if x[i] > x[j]:
                    temp = x[i]
                    x[i] = x[j]
                    x[j] = temp
                    id_ = idx[i]
                    idx[i] = idx[j]
                    idx[j] = id_

    def randomPermutations(self, perm, size):
        """
        perm : int list
        size : int
        Picks position for 1 to size at random and increments when value is already picked.
        Updates reference to perm 
        """
        index = [None] * size
        flag = [None] * size
        for n in range(size):
            index[n] = n
            flag[n] = True

        num = 0
        while num < size:
            start = random.randint(0, size-1)
            while True:
                if flag[start]:
                    # Add position to order of permutation.
                    perm[num] = index[start]
                    flag[start] = False
                    num += 1
                    break
                else:
                    # Try next position.
                    if start == size - 1:
                        start = 0
                    else:
                        start += 1

    #######################################################################
    #######################################################################
