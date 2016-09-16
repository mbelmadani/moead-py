# moead-py
A Python implementation of the decomposition based multi-objective evolutionary algorithm (MOEA/D).

MOEA/D is described in the following publication:
Zhang, Q. & Li, H. MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition. IEEE Trans. Evol. Comput. 11, 712–731 (2007).

## Description
The code in moead.py is a port of the MOEA/D algorithm provided by jmetal.metaheuristics.moead.MOEAD.java from the JMetal Java multi-objective metaheuristic algorithm framework (http://jmetal.sourceforge.net/). The JMetal specific functions have been converted to the DEAP (Distributed Evolutionary Algorithms in Python, http://deap.readthedocs.io/en/master/) equivalence.

## What's in here?

***moead.py*** - The class implementing the MOEA/D algorithm. Once the class MOEAD has been initialized, the algorithm can be executed with the *execute()* method.

***knapsack.py*** - An example of the multi-objective knapsack optimization problem. The original code is borrowed from DEAP (http://deap.readthedocs.io/en/master/examples/ga_knapsack.html) with modifications to use *moead.py* and an added triple-objective variant of the problem where weight difference between neighbouring items is minimized. You can run the example with:
```
python knapsack.py <SEED> <OBJECTIVES>
```

Where <SEED> is an optional integer for randomized execution. <OBJECTIVES> is either 2 or 3 and selects either the original 2 objective knapsack problem or a triple-objective variant.

## Status

The current version works with 2 or 3 objectives and more than 3 objectives if a weight file is provided. The algorithm has been tested on the knapsack examples (knapsack.py) provided above.

## Support and contributions

Contact Manuel Belmadani \<mbelm006@uottawa.ca\> for questions or comments. Pull requests are welcome! There's also the issues section (https://github.com/mbelmadani/moead-py/issues) where you can file bugs or request enhancements.
