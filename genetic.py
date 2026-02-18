rom common import Container, Vessel, SlotCoord, calculate_gm, MIN_GM
from deap import base, creator, tools  
from typing import List, Tuple, Any, Optional, cast, Dict
import random
import array
import multiprocessing
import copy


global_containers: List[Container] = []
global_id_map: Dict[int, int] = {} 


global_worker_vessel: Optional[Vessel] = None 
global_num_rows: int = 0


def init_worker(containers: List[Container], template_vessel: Vessel) -> None:
    """
    Worker initializer. 
    Creates a LOCAL DEEPCOPY of the vessel once when the process starts.
    We will reuse this object for every evaluation by clearing it.
    """
    global global_containers, global_id_map, global_worker_vessel, global_num_rows
    global_containers = containers
    global_id_map = {c.id: i for i, c in enumerate(containers)}
    
    
    
    global_worker_vessel = copy.deepcopy(template_vessel)
    global_num_rows = template_vessel.rows


def slots_from_column_index(col_idx: int, num_rows: int) -> Tuple[int, int]:
    """Decodes a linear Column Index into (Bay, Row)."""
    b = col_idx // num_rows
    r = col_idx % num_rows
    return b, r


def index_from_column(b: int, r: int, num_rows: int) -> int:
    """Encodes (Bay, Row) into a linear Column Index."""
    return (b * num_rows) + r


def fill_worker_vessel(individual: List[int], containers: List[Container]) -> Vessel:
    """
    Clears and refills the process-local vessel.
    No memory allocation for slots/vessel = Ultra Fast.
    """
    vessel = global_worker_vessel
    if vessel is None:
        raise RuntimeError("Worker not initialized")

    
    
    vessel.clear() 

    num_bays = vessel.bays
    num_rows = vessel.rows
    max_tiers = vessel.tiers
    total_cols = num_bays * num_rows
    
    
    col_heights = [[0 for _ in range(num_rows)] for _ in range(num_bays)]

    for i, container in enumerate(containers):
        preferred_col_idx = individual[i]

        
        for offset in range(total_cols):
            current_col_idx = (preferred_col_idx + offset) % total_cols
            b, r = slots_from_column_index(current_col_idx, num_rows)
            t = col_heights[b][r]

            if t < max_tiers:
                
                slot = vessel.get_slot_at(SlotCoord(b, r, t))
                if slot:
                    vessel.place(container, slot)
                    col_heights[b][r] += 1
                    break

    return vessel


def evaluate_stowage_worker(individual: Any) -> Tuple[float, float, float]:
    """Evaluation function for worker processes"""
    ind_as_list = cast(List[int], individual)

    
    vessel = fill_worker_vessel(ind_as_list, global_containers)

    
    gm = calculate_gm(vessel)
    safety_cost = (MIN_GM - gm) * 100.0 if gm < MIN_GM else 0.0

    
    rehandles = vessel.calculate_rehandles()

    
    balance_cost = abs(vessel.calculate_bay_moment()) + \
        abs(vessel.calculate_row_moment())

    return float(rehandles), float(safety_cost), float(balance_cost)



def build_vessel_clone(individual: List[int], containers: List[Container], template_vessel: Vessel) -> Vessel:
    """
    SLOW version for Local Search / Final Result.
    Must use deepcopy because we can't dirty the template.
    """
    vessel = copy.deepcopy(template_vessel)
    vessel.clear()
    
    num_bays = vessel.bays
    num_rows = vessel.rows
    max_tiers = vessel.tiers
    col_heights = [[0 for _ in range(num_rows)] for _ in range(num_bays)]
    total_cols = num_bays * num_rows

    for i, container in enumerate(containers):
        preferred_col_idx = individual[i]
        for offset in range(total_cols):
            current_col_idx = (preferred_col_idx + offset) % total_cols
            b, r = slots_from_column_index(current_col_idx, num_rows)
            t = col_heights[b][r]

            if t < max_tiers:
                slot = vessel.get_slot_at(SlotCoord(b, r, t))
                if slot:
                    vessel.place(container, slot)
                    col_heights[b][r] += 1
                    break
    return vessel


def evaluate_local(individual: Any, containers: List[Container], template_vessel: Vessel) -> Tuple[float, float, float]:
    """Evaluation function for Main Process (uses deepcopy safely)"""
    ind_as_list = cast(List[int], individual)
    
    vessel = build_vessel_clone(ind_as_list, containers, template_vessel)

    gm = calculate_gm(vessel)
    safety_cost = (MIN_GM - gm) * 100.0 if gm < MIN_GM else 0.0
    rehandles = vessel.calculate_rehandles()
    balance_cost = abs(vessel.calculate_bay_moment()) + abs(vessel.calculate_row_moment())

    return float(rehandles), float(safety_cost), float(balance_cost)


def apply_memetic_local_search(individual: Any, containers: List[Container], template_vessel: Vessel) -> bool:
    """
    Bay-balance-aware local search.
    """
    ind_list = cast(List[int], individual)
    modified = False
    
    num_bays = template_vessel.bays
    num_rows = template_vessel.rows

    
    bay_weights = [0.0] * num_bays
    for i, col_idx in enumerate(ind_list):
        b, _ = slots_from_column_index(col_idx, num_rows)
        bay_weights[b] += containers[i].weight

    total_weight = sum(bay_weights) or 1.0
    center_bay = (num_bays - 1) / 2.0
    current_moment = sum(bay_weights[b] * (b - center_bay)
                         for b in range(num_bays)) / total_weight

    if abs(current_moment) > 0.5:
        heavy_side_start = 0 if current_moment < 0 else int(center_bay) + 1
        heavy_side_end = int(center_bay) + \
            1 if current_moment < 0 else num_bays

        target_side_start = int(center_bay) + 1 if current_moment < 0 else 0
        target_side_end = num_bays if current_moment < 0 else int(
            center_bay) + 1

        candidates = [i for i, col in enumerate(ind_list)
                      if heavy_side_start <= slots_from_column_index(col, num_rows)[0] < heavy_side_end]

        if candidates:
            idx_to_move = random.choice(candidates)
            new_b = random.randint(target_side_start, max(
                target_side_start, target_side_end - 1))
            new_r = random.randint(0, num_rows - 1)
            ind_list[idx_to_move] = index_from_column(new_b, new_r, num_rows)
            modified = True

    return modified


def setup_toolbox(containers: List[Container], vessel: Vessel) -> base.Toolbox:
    num_containers = len(containers)
    total_columns = vessel.bays * vessel.rows

    if hasattr(creator, "FitnessMin"):
        del creator.FitnessMin
    if hasattr(creator, "Individual"):
        del creator.Individual

    
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
    creator.create("Individual", array.array, typecode='i',
                   fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, 0, total_columns - 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_int, num_containers) 
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mutate", tools.mutUniformInt, low=0,
                     up=total_columns-1, indpb=0.05)
    return toolbox


def solve_stowage_genetic(containers: List[Container], vessel: Vessel) -> Vessel:
    
    
    
    
    sorted_containers = sorted(
        containers,
        key=lambda k: (k.dischargePort, k.weight),
        reverse=True
    )

    num_bays = vessel.bays
    num_rows = vessel.rows
    max_tiers = vessel.tiers

    N = len(sorted_containers)
    pop_size = min(200, max(50, 4 * N))
    pop_size = pop_size - (pop_size % 4)
    n_gen = min(100, max(30, N))

    toolbox = setup_toolbox(sorted_containers, vessel)
    toolbox.register("evaluate", evaluate_stowage_worker)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("select", tools.selNSGA2)

    cpu_count = multiprocessing.cpu_count()
    pop = toolbox.population(n=pop_size) 

    
    num_seeds = max(2, int(pop_size * 0.25))
    total_cols = num_bays * num_rows
    smart_genome = [0] * N
    curr_col = 0
    curr_h = 0
    for i in range(N):
        smart_genome[i] = curr_col
        curr_h += 1
        if curr_h >= max_tiers:
            curr_col = (curr_col + 1) % total_cols
            curr_h = 0
    for i in range(num_seeds):
        pop[i] = creator.Individual(smart_genome) 

    
    fitnesses = [evaluate_local(
        ind, sorted_containers, vessel) for ind in pop]
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    tools.emo.assignCrowdingDist(pop)

    
    with multiprocessing.Pool(
        processes=cpu_count,
        initializer=init_worker,
        initargs=(sorted_containers, vessel) 
    ) as pool:
        toolbox.register("map", pool.map)

        for gen in range(1, n_gen + 1):
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring]

            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.7:
                    toolbox.mate(ind1, ind2) 
                    del ind1.fitness.values
                    del ind2.fitness.values

            for ind in offspring:
                if random.random() < 0.1:
                    toolbox.mutate(ind) 
                    del ind.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(
                toolbox.evaluate, invalid_ind) 
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            
            if gen % 10 == 0:
                valid_offspring = [ind for ind in offspring if ind.fitness.valid]
                valid_offspring.sort(key=lambda x: sum(x.fitness.values))
                for ind in valid_offspring[:max(1, int(len(valid_offspring) * 0.05))]:
                    if apply_memetic_local_search(ind, sorted_containers, vessel):
                        del ind.fitness.values

            invalid_after_ls = [ind for ind in offspring if not ind.fitness.valid]
            if invalid_after_ls:
                ls_fitnesses = toolbox.map(toolbox.evaluate, invalid_after_ls) 
                for ind, fit in zip(invalid_after_ls, ls_fitnesses):
                    ind.fitness.values = fit

            pop = toolbox.select(pop + offspring, pop_size) 

    
    pareto_fronts = tools.sortNondominated(
        pop, len(pop), first_front_only=True)
    
    
    best_ind = min(
        pareto_fronts[0], key=lambda ind: ind.fitness.values[1] * 1000 + ind.fitness.values[0])

    
    best_vessel = build_vessel_clone(
        cast(List[int], best_ind),
        sorted_containers, vessel
    )
    return best_vessel
