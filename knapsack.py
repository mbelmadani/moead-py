#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import random
import numpy
import sys

from moead import MOEAD

from deap import base
from deap import creator
from deap import tools

IND_INIT_SIZE = 5
MAX_ITEM = 50
MAX_WEIGHT = 50
NBR_ITEMS = 20


NGEN = 50
MU = 50
LAMBDA = 2
CXPB = 0.7
MUTPB = 0.2

# Create random items and store them in the items' dictionary.
items = {}
for i in range(NBR_ITEMS):
    items[i] = (random.randint(1, 10), random.uniform(0, 100))


def evalKnapsack(individual):    
    weight = 0.0
    value = 0.0
    for item in individual:
        weight += items[item][0]
        value += items[item][1]
    if len(individual) > MAX_ITEM or weight > MAX_WEIGHT:
        return 1e30, 0.0 # Ensure overweighted bags are dominated
    return weight, value

def evalKnapsackBalanced(individual):
    """
    Variant of the original weight-value knapsack problem with added third object being minimizing weight difference between items.
    """
    weight, value = evalKnapsack(individual)
    balance = 0.0
    for a,b in zip(individual, list(individual)[1:]):
        balance += abs(items[a][0]-items[b][0])
    if len(individual) > MAX_ITEM or weight > MAX_WEIGHT:
        return weight, value, 1e30 # Ensure overweighted bags are dominated
    return weight, value, balance

def cxSet(ind1, ind2):
    """Apply a crossover operation on input sets. The first child is the
    intersection of the two sets, the second child is the difference of the
    two sets.
    """
    temp = set(ind1)                # Used in order to keep type
    ind1 &= ind2                    # Intersection (inplace)
    ind2 ^= temp                    # Symmetric Difference (inplace)
    return ind1, ind2
    
def mutSet(individual):
    """Mutation that pops or add an element."""
    if random.random() < 0.5:
        if len(individual) > 0:     # We cannot pop from an empty set
            individual.remove(random.choice(sorted(tuple(individual))))
    else:
        individual.add(random.randrange(NBR_ITEMS))
    return individual,

def main(objectives=2, seed=64):
    random.seed(seed)

    # Create the item dictionary: item name is an integer, and value is 
    # a (weight, value) 2-uple.
    if objectives == 2:
        creator.create("Fitness", base.Fitness, weights=(-1.0, 1.0))
    elif objectives == 3:
        creator.create("Fitness", base.Fitness, weights=(-1.0, 1.0, -1.0))
    else:
        print "No evaluation function available for", objectives, "objectives."
        sys.exit(-1)

        
    creator.create("Individual", set, fitness=creator.Fitness)

    toolbox = base.Toolbox()

    # Attribute generator
    toolbox.register("attr_item", random.randrange, NBR_ITEMS)

    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                     toolbox.attr_item, IND_INIT_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    if objectives == 2:
        toolbox.register("evaluate", evalKnapsack)
    elif objectives == 3:
        toolbox.register("evaluate", evalKnapsackBalanced)
    else:
        print "No evaluation function available for", objectives, "objectives."
        sys.exit(-1)
        

    toolbox.register("mate", cxSet)
    toolbox.register("mutate", mutSet)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=MU)
    hof = tools.ParetoFront()

    stats = {}
    def lambda_factory(idx):
        return lambda ind: ind.fitness.values[idx]                

    fitness_tags = ["Weight", "Value"]
    for tag in fitness_tags:
        s = tools.Statistics( key=lambda_factory(
                    fitness_tags.index(tag)
                ))
        stats[tag] = s

    mstats = tools.MultiStatistics(**stats)
    mstats.register("avg", numpy.mean, axis=0)
    mstats.register("std", numpy.std, axis=0)
    mstats.register("min", numpy.min, axis=0)
    mstats.register("max", numpy.max, axis=0)

    ea = MOEAD(pop, toolbox, MU, CXPB, MUTPB, ngen=NGEN, stats=mstats, halloffame=hof, nr=LAMBDA)
    pop = ea.execute()
    
    return pop, stats, hof
                 
if __name__ == "__main__":
    objectives = 2
    seed = 64
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
    if len(sys.argv) > 2:
        objectives = int(sys.argv[2])

    pop,stats,hof = main(objectives)

    pop = [str(p) +" "+ str(p.fitness.values) for p in pop]
    hof = [str(h) +" "+ str(h.fitness.values) for h in hof]
    print "POP:"
    print "\n".join(pop)

    print "PF:"
    print "\n".join(hof)
