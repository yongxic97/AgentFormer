import numpy as np

from copy import copy

class Individual:
    def __init__(self, genotype_length):
        self.genotype = np.zeros(genotype_length)
        self.fitness = None

    @staticmethod
    def initializeWithGenotype(genotype: np.array):
        individual = Individual(len(genotype))
        individual.genotype = genotype.copy()
        return individual
    
    @staticmethod
    def initializeUniformAtRandom(genotype_length):
        individual = Individual(genotype_length)
        individual.genotype = np.random.choice((0,1), p=(0.5, 0.5), size=genotype_length)
        return individual

def elitist_selection( population, offspring ):
    selection = []
    # your code here
    desire_num = len(population)
    rolled_out = 0
    candidate_os_list = np.arange(len(offspring))
    while rolled_out < desire_num:
        os1 = np.random.choice(candidate_os_list, 1)[0]
        
        parents = offspring[os1].parent_idxs
        parent1, parent2 = population[parents[0]], population[parents[1]]
        os2 = -1
        for i in candidate_os_list:
            if (not np.array_equal(offspring[i].genotype, offspring[os1].genotype)) and offspring[i].parent_idxs == offspring[os1].parent_idxs:
                os2 = i
        fitnesses = np.array([offspring[os1].fitness, offspring[os2].fitness, parent1.fitness, parent2.fitness])
        largests = np.argsort(-fitnesses)
        winner = largests[0]
        runner_up = largests[1]
        print(winner, runner_up)
        # winner = np.argmax(fitnesses)
        rolled_out += 2
        if winner==0:
            selection.append(offspring[os1])
        elif winner==1:
            selection.append(offspring[os2])
        elif winner==2:
            selection.append(parent1)
        elif winner==3:
            selection.append(parent2)

        if runner_up==0:
            selection.append(offspring[os1])
        elif runner_up==1:
            selection.append(offspring[os2])
        elif runner_up==2:
            selection.append(parent1)
        elif runner_up==3:
            selection.append(parent2)
        # print(candidate_os_list)
        print(len(selection))
        candidate_os_list = np.setdiff1d(candidate_os_list, os1)
        candidate_os_list = np.setdiff1d(candidate_os_list, os2)

    return selection

def evaluate( individual: Individual ):
    # Example evaluation function
    # Solutions provided to the selection function are already evaluated
    # and their fitness is cached in `individual.fitness`.
    # You should not apply evaluate to individuals in your solution.
    individual.fitness = np.sum(individual.genotype)

def create_offspring(l, parent_a, parent_b, parent_idx_a, parent_idx_b):
    # Normally this function would perform crossover & mutation.
    # This is just a placeholder!
    offspring_a = Individual.initializeUniformAtRandom(l)
    offspring_b = Individual.initializeUniformAtRandom(l)

    offspring_a.parent_idxs = [parent_idx_a, parent_idx_b]
    offspring_b.parent_idxs = [parent_idx_a, parent_idx_b]

    evaluate( offspring_a )
    evaluate( offspring_b )
    return [offspring_a, offspring_b]

def test_example():
    l = 5
    pop_size = 20
    pop = [Individual.initializeUniformAtRandom(l) for i in range(pop_size)]

    # solutions provided to the selection function are already evaluated
    # and their fitness is cached in `individual.fitness`.
    for individual in pop:
        evaluate( individual )

    offspring = []
    for i in range(0, pop_size, 2):
        offspring += create_offspring(l, pop[i], pop[i + 1], i, i + 1)


    pop_f = [p.fitness for p in pop]
    print(f"from population {pop_f}")
    offspring_f = [o.fitness for o in offspring]
    print(f"and offspring {offspring_f}")

    # perform selection
    selection = elitist_selection(pop, offspring)

    selection_f = [s.fitness for s in selection]
    print(f"selected {selection_f}")

test_example()


# ###################
import numpy as np
from library import Individual

def tournament_selection( population, offspring ):
    selection_pool = np.concatenate((population, offspring),axis=None)
    tournament_size = 4
    selection = []
    assert len(selection_pool) % tournament_size == 0, "Population size should be a multiple of tournament size"

    # Your code here
    candidate_indices = np.arange(len(selection_pool))
    desire_size = len(population)
    rolled_out = 0
    while rolled_out<desire_size:
        t_indices=np.random.choice(candidate_indices, tournament_size,replace=False) # tournament indices
        fitnesses = []
        for i in range(tournament_size):
            fitnesses.append(selection_pool[t_indices[i]].fitness)
        winner = np.argmax(fitnesses)
        # candidate_indices = np.delete(candidate_indices, winner)
        # if selection_pool[t_indices[winner]] in selection:
        #     # selection_pool = np.delete(selection_pool, t_indices[winner])
        #     pass
        # else:
        rolled_out += 1
        selection.append(selection_pool[t_indices[winner]])

    return selection