from common import Cont, CostReport, Ship
from deap import base, creator, tools, algorithms
from typing import List, Tuple
import random
import array
import multiprocessing

# --- Global variables for Worker Processes ---
# These are only set inside the worker processes
global_containers = None
global_num_bays = None
global_num_rows = None
global_max_tiers = None

def init_worker(containers, num_bays, num_rows, max_tiers):
    """
    Initializes the worker process with the read-only data.
    This runs once per CPU core, not once per evaluation.
    """
    global global_containers, global_num_bays, global_num_rows, global_max_tiers
    global_containers = containers
    global_num_bays = num_bays
    global_num_rows = num_rows
    global_max_tiers = max_tiers

def slots_from_index(idx: int, num_bays: int, num_rows: int, max_tiers: int) -> Tuple[int, int, int]:
    per_bay = num_rows * max_tiers
    b = idx // per_bay
    rem = idx % per_bay
    r = rem // max_tiers
    t = rem % max_tiers
    return b, r, t

def build_ship_from_individual(individual: List[int], containers: List[Cont], num_bays: int, num_rows: int, max_tiers: int):
    ship: Ship = [[[None for _ in range(max_tiers)] for _ in range(num_rows)] for _ in range(num_bays)]
    total_slots = num_bays * num_rows * max_tiers

    for i, container in enumerate(containers):
        preferred_idx = individual[i]
        placed = False
        
        for offset in range(total_slots):
            current_idx = (preferred_idx + offset) % total_slots
            b, r, t = slots_from_index(current_idx, num_bays, num_rows, max_tiers)
            
            if b >= num_bays or r >= num_rows or t >= max_tiers:
                continue

            if t > 0:
                if ship[b][r][t-1] is None:
                    continue 

            if ship[b][r][t] is None:
                ship[b][r][t] = container
                placed = True
                break
    return ship

def evaluate_stowage_worker(individual):
    """
    Evaluation function optimized for multiprocessing.
    It uses the global variables set by init_worker.
    """
    # Use the global data stored in this worker process
    ship = build_ship_from_individual(
        individual, 
        global_containers, 
        global_num_bays, 
        global_num_rows, 
        global_max_tiers
    )
    
    cost = CostReport(ship)
    rehandles = cost.rehandles
    b_mom = abs(cost.bay_moment)
    r_mom = abs(cost.row_moment)
    t_mom = abs(cost.tier_moment)
    return float(rehandles), float(b_mom), float(r_mom), float(t_mom)

def setup_toolbox(num_containers: int, num_bays: int, num_rows: int, max_tiers: int):
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-50.0, -1.0, -1.0, -1.0))

    if not hasattr(creator, "Individual"):
        creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    total_slots = num_bays * num_rows * max_tiers
    toolbox.register("attr_int", random.randint, 0, total_slots - 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, num_containers)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    return toolbox

def solve_stowage_genetic(containers: List[Cont], num_bays: int, num_rows: int, max_tiers: int):
    N = len(containers)
    
    pop_size = min(400, max(40, 5 * N))
    n_gen = min(1000, max(80, 5 * N))
    
    toolbox = setup_toolbox(N, num_bays, num_rows, max_tiers)
    
    # NOTE: We do NOT pass containers here anymore. The worker already has them via initializer.
    toolbox.register("evaluate", evaluate_stowage_worker)
    
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=(num_bays*num_rows*max_tiers)-1, indpb=0.05)
    toolbox.register("select", tools.selNSGA2)

    # Determine core count explicitly
    cpu_count = multiprocessing.cpu_count()

    # Initialize Pool with global data
    # This ensures 'containers' is only copied once per core, not once per individual
    with multiprocessing.Pool(
        processes=cpu_count, 
        initializer=init_worker, 
        initargs=(containers, num_bays, num_rows, max_tiers)
    ) as pool:
        
        toolbox.register("map", pool.map)

        pop = toolbox.population(n=pop_size)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", lambda x: tuple(map(min, zip(*x)))) 
        stats.register("avg", lambda x: tuple(map(lambda y: sum(y)/len(y), zip(*x))))

        pop, logbook = algorithms.eaMuPlusLambda(
            pop, toolbox, 
            mu=pop_size, lambda_=pop_size, 
            cxpb=0.6, mutpb=0.4, 
            ngen=n_gen, stats=stats, verbose=False
        )

        pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]

        best_ind = min(pareto_front, key=lambda ind: (ind.fitness.values[0], sum(ind.fitness.values[1:])))
        
        # Build final result locally
        best_ship = build_ship_from_individual(best_ind, containers, num_bays, num_rows, max_tiers)

        return best_ship