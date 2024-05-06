import numpy as np
from library import Individual

def tournament_selection( population, offspring ):
    selection_pool = np.concatenate((population, offspring),axis=None)
    tournament_size = 4
    selection = []
    assert len(selection_pool) % tournament_size == 0, "Population size should be a multiple of tournament size"

    # Your code here
    while len(selection_pool) >= tournament_size:
        np.random.shuffle(selection_pool)
        for i in range(0,len(selection_pool), tournament_size):
            tournament = selection_pool[i,i+tournament_size]
            tournament_fitnesses = [individual.fitness for individual in tournament]
            winner_index = np.argmax(tournament_fitnesses)
            winner = tournament[winner_index]
            selection.append(winner)
            if len(selection) == len(population):
                print("max count reached.")
                break
        selection_pool = np.setdiff1d(selection_pool, selection)
    return selection
# import numpy as np
# from library import Individual

# def tournament_selection( population, offspring ):
#     selection_pool = np.concatenate((population, offspring),axis=None)
#     tournament_size = 4
#     selection = []
#     assert len(selection_pool) % tournament_size == 0, "Population size should be a multiple of tournament size"

#     # Your code here
#     desire_size = len(population)
#     rolled_out = 0
#     while rolled_out<desire_size:
#         # np.random.shuffle(selection_pool)
#         # t_indices = np.array([0,1,2,3])
#         # # t_indices=np.random.choice(len(selection_pool),tournament_size,replace=False) # tournament indices
#         # # fitnesses = []
#         # # for i in range(tournament_size):
#         #     # fitnesses.append(selection_pool[t_indices[i]].fitness)
#         #     # np.delete(selection_pool[t_indices[i]])
#         # fitnesses = [selection_pool[i].fitness for i in t_indices]
#         # # semi-final
#         # candidate1 = 0 if fitnesses[0] > fitnesses[1] else 1
#         # candidate2 = 2 if fitnesses[2] > fitnesses[3] else 3
#         # # final
#         # winner = candidate1 if fitnesses[candidate1] > fitnesses[candidate2] else candidate2

#         # rolled_out += 1
#         # selection.append(selection_pool[t_indices[winner]])
        
#         # if len(selection_pool) - 4 >= 4:
#         #     selection_pool = np.delete(selection_pool,t_indices)
#         # else:
#         #     # continue
#         #     num_to_delete = len(selection_pool) - 4
#         #     for i in range(num_to_delete):
#         #         selection_pool = np.delete(selection_pool, t_indices[i])
#         ########################
#         np.random.shuffle(selection_pool)
#         this_round = len(selection_pool) // tournament_size if desire_size-rolled_out > (len(selection_pool) // tournament_size) else desire_size-rolled_out
#         print(this_round)
#         # this_round = len(selection_pool) // tournament_size
#         fitnesses = [selection_pool[i].fitness for i in range(len(selection_pool))]
#         candidate1 = np.zeros(this_round)
#         candidate2 = np.zeros(this_round)
#         winner = np.zeros(this_round)
#         for i in range(this_round):
#             # semi-final
#             candidate1[i] = 0 if fitnesses[tournament_size*i+0] > fitnesses[tournament_size*i+1] else 1
#             candidate2[i] = 2 if fitnesses[tournament_size*i+0] > fitnesses[tournament_size*i+1] else 3
#             # final
#             winner[i] = candidate1[i] if fitnesses[int(tournament_size*i+candidate1[i])] > fitnesses[int(tournament_size*i+candidate2[i])] else candidate2[i]
#             selection.append(selection_pool[int(tournament_size*i+winner[i])])
#             selection_pool = np.delete(selection_pool, int(tournament_size*i+winner[i]))

#         rolled_out += this_round
            
#     return selection