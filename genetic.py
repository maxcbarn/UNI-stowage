from common import Container, CostReport, Vessel, SlotCoord, calculate_gm,  MIN_GM
from deap import base, creator, tools  # type: ignore
from typing import List, Tuple, Any, Optional, cast, Dict
import random
import array
import multiprocessing
import copy

# --- Global variables for Worker Processes ---
global_containers: List[Container] = []
global_id_map: Dict[int, int] = {}  # Optimization: O(1) lookup map
global_num_bays: int = 0
global_num_rows: int = 0
global_max_tiers: int = 0


def init_worker(containers: List[Container], num_bays: int, num_rows: int, max_tiers: int) -> None:
    global global_containers, global_id_map, global_num_bays, global_num_rows, global_max_tiers
    global_containers = containers
    # OPTIMIZATION: Build ID map once
    global_id_map = {c.id: i for i, c in enumerate(containers)}
    global_num_bays = num_bays
    global_num_rows = num_rows
    global_max_tiers = max_tiers


def slots_from_column_index(col_idx: int, num_rows: int) -> Tuple[int, int]:
    """Decodes a linear Column Index into (Bay, Row)."""
    b = col_idx // num_rows
    r = col_idx % num_rows
    return b, r


def index_from_column(b: int, r: int, num_rows: int) -> int:
    """Encodes (Bay, Row) into a linear Column Index."""
    return (b * num_rows) + r


def build_vessel_from_individual(individual: List[int], containers: List[Container], num_bays: int, num_rows: int, max_tiers: int) -> Vessel:
    """
    Decodes a GA individual (list of column preferences) into a Vessel object.
    Operates directly on the Vessel class.
    """
    vessel = Vessel(num_bays, num_rows, max_tiers)

    # Track current height of each column to avoid scanning
    col_heights = [[0 for _ in range(num_rows)] for _ in range(num_bays)]
    total_cols = num_bays * num_rows

    for i, container in enumerate(containers):
        preferred_col_idx = individual[i]
        placed = False

        # Linear probe for the first available column
        for offset in range(total_cols):
            current_col_idx = (preferred_col_idx + offset) % total_cols
            b, r = slots_from_column_index(current_col_idx, num_rows)
            t = col_heights[b][r]

            # Only check capacity. Blocking is impossible due to pre-sort.
            if t < max_tiers:
                slot = vessel.get_slot_at(SlotCoord(b, r, t))
                if slot:
                    vessel.place(container, slot)
                    col_heights[b][r] += 1
                    placed = True
                    break

        # Note: Leftovers are implicitly handled by the cost function (container not in vessel)

    return vessel


def evaluate_stowage_worker(individual: Any) -> Tuple[float, float, float]:
    """Evaluation function for worker processes (uses global variables)"""
    ind_as_list = cast(List[int], individual)

    vessel = build_vessel_from_individual(
        ind_as_list,
        global_containers,
        global_num_bays,
        global_num_rows,
        global_max_tiers
    )

    # Use the UNIFIED Cost Function
    # We return a tuple for Multi-Objective Optimization (NSGA-II)
    # Objective 1: Efficiency (Rehandles)
    # Objective 2: Safety (GM Penalty) - Must be minimized!
    # Objective 3: Balance (Bay/Row Moments)

    # 1. Safety (GM)
    gm = calculate_gm(vessel)
    safety_cost = (MIN_GM - gm) * 100.0 if gm < MIN_GM else 0.0  # Scale for GA

    # 2. Efficiency
    # Convert to ship list for the fast RehandlesNumber function in common
    rehandles = vessel.calculate_rehandles()

    # 3. Balance
    balance_cost = abs(vessel.calculate_bay_moment()) + \
        abs(vessel.calculate_row_moment())

    return float(rehandles), float(safety_cost), float(balance_cost)


def evaluate_local(individual: Any, containers: List[Container], num_bays: int, num_rows: int, max_tiers: int) -> Tuple[float, float, float]:
    """Evaluation function for non-pool contexts"""
    ind_as_list = cast(List[int], individual)
    vessel = build_vessel_from_individual(
        ind_as_list, containers, num_bays, num_rows, max_tiers)

    gm = calculate_gm(vessel)
    safety_cost = (MIN_GM - gm) * 100.0 if gm < MIN_GM else 0.0

    rehandles = vessel.calculate_rehandles()
    balance_cost = abs(vessel.calculate_bay_moment()) + \
        abs(vessel.calculate_row_moment())

    return float(rehandles), float(safety_cost), float(balance_cost)


def apply_memetic_local_search(individual: Any, containers: List[Container], num_bays: int, num_rows: int, max_tiers: int) -> bool:
    """
    Bay-balance-aware local search.
    Adapted to use Vessel logic indirectly via column indices.
    """
    ind_list = cast(List[int], individual)
    modified = False

    # We rely on the fact that `build_vessel` places containers in the column
    # specified by the gene (or the next available one).
    # Heuristic: Calculate current approximate bay weights based on gene targets.

    bay_weights = [0.0] * num_bays
    for i, col_idx in enumerate(ind_list):
        b, _ = slots_from_column_index(col_idx, num_rows)
        # Note: This is an approximation. If the column is full, the container moves.
        # But for mutation guidance, it's accurate enough.
        bay_weights[b] += containers[i].weight

    total_weight = sum(bay_weights) or 1.0
    center_bay = (num_bays - 1) / 2.0
    current_moment = sum(bay_weights[b] * (b - center_bay)
                         for b in range(num_bays)) / total_weight

    # Simple Hill Climbing: If moment is bad, move a random container
    # from the heavy side to the light side.
    if abs(current_moment) > 0.5:
        heavy_side_start = 0 if current_moment < 0 else int(center_bay) + 1
        heavy_side_end = int(center_bay) + \
            1 if current_moment < 0 else num_bays

        target_side_start = int(center_bay) + 1 if current_moment < 0 else 0
        target_side_end = num_bays if current_moment < 0 else int(
            center_bay) + 1

        # Pick a random container that is currently targeted at the heavy side
        candidates = [i for i, col in enumerate(ind_list)
                      if heavy_side_start <= slots_from_column_index(col, num_rows)[0] < heavy_side_end]

        if candidates:
            idx_to_move = random.choice(candidates)
            # Move to random column in target side
            new_b = random.randint(target_side_start, max(
                target_side_start, target_side_end - 1))
            new_r = random.randint(0, num_rows - 1)
            ind_list[idx_to_move] = index_from_column(new_b, new_r, num_rows)
            modified = True

    return modified


def setup_toolbox(containers: List[Container], num_bays: int, num_rows: int, max_tiers: int) -> base.Toolbox:
    num_containers = len(containers)

    if hasattr(creator, "FitnessMin"):
        del creator.FitnessMin
    if hasattr(creator, "Individual"):
        del creator.Individual

    # Minimize: Rehandles, Safety Penalty, Balance Penalty
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
    creator.create("Individual", array.array, typecode='i',
                   fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    total_columns = num_bays * num_rows
    toolbox.register("attr_int", random.randint, 0, total_columns - 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_int, num_containers)  # type: ignore
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register mutation (using our approximated logic)
    toolbox.register("mutate", tools.mutUniformInt, low=0,
                     up=total_columns-1, indpb=0.05)

    return toolbox


def solve_stowage_genetic(containers: List[Container], vessel: Vessel) -> Vessel:
    # 1. GLOBAL SORT (Crucial for Heuristic Decoder)
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
    n_gen = min(100, max(30, N))  # Reduced gens for speed test

    toolbox = setup_toolbox(
        sorted_containers, num_bays, num_rows, max_tiers)
    toolbox.register("evaluate", evaluate_stowage_worker)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("select", tools.selNSGA2)

    cpu_count = multiprocessing.cpu_count()
    pop = toolbox.population(n=pop_size)  # type: ignore

    # Seed with heuristic (snake fill)
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
        pop[i] = creator.Individual(smart_genome)  # type: ignore

    # Initial Eval
    fitnesses = [evaluate_local(
        ind, sorted_containers, num_bays, num_rows, max_tiers) for ind in pop]
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    tools.emo.assignCrowdingDist(pop)

    # Evolution
    with multiprocessing.Pool(
        processes=cpu_count,
        initializer=init_worker,
        initargs=(sorted_containers, num_bays, num_rows, max_tiers)
    ) as pool:
        toolbox.register("map", pool.map)

        for gen in range(1, n_gen + 1):
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring]

            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.7:
                    toolbox.mate(ind1, ind2)  # type: ignore
                    del ind1.fitness.values
                    del ind2.fitness.values

            for ind in offspring:
                if random.random() < 0.1:
                    toolbox.mutate(ind)  # type: ignore
                    del ind.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(
                toolbox.evaluate, invalid_ind)  # type: ignore
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Local Search (Every 10 gens)
            if gen % 10 == 0:
                offspring.sort(key=lambda x: sum(x.fitness.values))
                for i in range(int(len(offspring) * 0.05)):
                    if apply_memetic_local_search(offspring[i], sorted_containers, num_bays, num_rows, max_tiers):
                        del offspring[i].fitness.values  # Re-eval next loop

            pop = toolbox.select(pop + offspring, pop_size)  # type: ignore

    # Best Result
    pareto_fronts = tools.sortNondominated(
        pop, len(pop), first_front_only=True)
    # Weight objectives: Safety (1) > Rehandles (0) > Balance (2)
    # But since we minimize, we just look for min sum or prioritized sort
    best_ind = min(
        pareto_fronts[0], key=lambda ind: ind.fitness.values[1] * 1000 + ind.fitness.values[0])

    best_vessel = build_vessel_from_individual(
        cast(List[int], best_ind),
        sorted_containers, num_bays, num_rows, max_tiers
    )
    return best_vessel
