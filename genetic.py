from common import Cont, CostReport, Ship
from deap import base, creator, tools, algorithms
from typing import List, Tuple
import random
import array


def slots_from_index(idx: int, num_bays: int, num_rows: int, max_tiers: int) -> Tuple[int, int, int]:
    # Decodes a linear index back to (bay, row, tier)
    per_bay = num_rows * max_tiers
    b = idx // per_bay
    rem = idx % per_bay
    r = rem // max_tiers
    t = rem % max_tiers
    return b, r, t


def build_ship_from_individual(individual: List[int], containers: List[Cont], num_bays: int, num_rows: int, max_tiers: int):
    ship: Ship = [[[None for _ in range(max_tiers)] for _ in range(num_rows)] for _ in range(num_bays)]

    total_slots = num_bays * num_rows * max_tiers
    occupied = [False] * total_slots

    # Determine order: The individual defines the PREFERRED slot for each container
    for i, container in enumerate(containers):
        preferred_idx = individual[i]
        placed = False
        
        for offset in range(total_slots):
            current_idx = (preferred_idx + offset) % total_slots

            b, r, t = slots_from_index(current_idx, num_bays, num_rows, max_tiers)
            
            if b >= num_bays or r >= num_rows or t >= max_tiers:
                continue

            # Floating rule
            if t > 0:
                if ship[b][r][t-1] is None:
                    continue 

            if ship[b][r][t] is None:
                ship[b][r][t] = container
                placed = True
                break

        if not placed:
            pass

    return ship

def setup_toolbox(num_containers: int, num_bays: int, num_rows: int, max_tiers: int):
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0))

    if not hasattr(creator, "Individual"):
        creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    total_slots = num_bays * num_rows * max_tiers
    toolbox.register("attr_int", random.randint, 0, total_slots - 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, num_containers)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # The individuals are a array of positions of the ship, each index in this array represents the same index in the array of containers.

    return toolbox

def evaluate_stowage(individual, containers, num_bays, num_rows, max_tiers):
    ship = build_ship_from_individual( individual, containers, num_bays, num_rows, max_tiers )
    cost = CostReport(ship)
    rehandles = cost.rehandles
    b_mom = abs(cost.bay_moment)
    r_mom = abs(cost.row_moment)
    t_mom = abs(cost.tier_moment)
    return float(rehandles), float(b_mom), float(r_mom), float(t_mom)



def solve_stowage_genetic(containers: List[Cont], num_bays: int, num_rows: int, max_tiers: int,pop_size=100, n_gen=50):
    toolbox = setup_toolbox(len(containers), num_bays, num_rows, max_tiers)
    # Register Operators
    toolbox.register("evaluate", evaluate_stowage, containers=containers, num_bays=num_bays, num_rows=num_rows, max_tiers=max_tiers)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0,up=(num_bays*num_rows*max_tiers)-1, indpb=0.05)
    toolbox.register("select", tools.selNSGA2)
    # Selection NSGA-II is standard for Multi-Objective

    pop = toolbox.population(n=pop_size)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", lambda x: tuple(map(min, zip(*x)))) 
    stats.register("avg", lambda x: tuple(map(lambda y: sum(y)/len(y), zip(*x))))

    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=pop_size, lambda_=pop_size, cxpb=0.7, mutpb=0.2, ngen=n_gen, stats=stats, verbose=False)

    pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]

    # Select a single "best" solution for the return value
    # Strategy: Prioritize Rehandles (index 0), then sum of moments
    best_ind = min(pareto_front, key=lambda ind: ( ind.fitness.values[0], sum(ind.fitness.values[1:])))
    best_ship = build_ship_from_individual(best_ind, containers, num_bays, num_rows, max_tiers)

    return best_ship
