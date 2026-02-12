from common import Cont, CostReport, Ship
from deap import base, creator, tools  # type: ignore
from typing import List, Tuple, Any, Optional, cast, Dict
import random
import array
import multiprocessing

# --- Global variables for Worker Processes ---
global_containers: List[Cont] = []
global_id_map: Dict[int, int] = {}  # Optimization: O(1) lookup map
global_num_bays: int = 0
global_num_rows: int = 0
global_max_tiers: int = 0


def init_worker(containers: List[Cont], num_bays: int, num_rows: int, max_tiers: int) -> None:
    global global_containers, global_id_map, global_num_bays, global_num_rows, global_max_tiers
    global_containers = containers
    # OPTIMIZATION: Build ID map once to avoid O(N) scans in local search
    global_id_map = {c['id']: i for i, c in enumerate(containers)}
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


def build_ship_from_individual(individual: List[int], containers: List[Cont], num_bays: int, num_rows: int, max_tiers: int) -> Ship:
    """
    OPTIMIZED DECODER
    Since 'containers' is pre-sorted (Dest DESC, Weight DESC), we NO LONGER
    need to check for blocking or stability during placement.
    We simply fill the target column.
    """
    ship: Ship = [[[None for _ in range(max_tiers)]
                   for _ in range(num_rows)]
                  for _ in range(num_bays)]

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
                ship[b][r][t] = container
                col_heights[b][r] += 1
                placed = True
                break

        # Note: If not placed (ship full), it's ignored (CostReport handles leftovers)

    return ship


def evaluate_stowage_worker(individual: Any) -> Tuple[float, float, float]:
    """Evaluation function for worker processes (uses global variables)"""
    ind_as_list = cast(List[int], individual)

    ship = build_ship_from_individual(
        ind_as_list,
        global_containers,
        global_num_bays,
        global_num_rows,
        global_max_tiers
    )

    # REMOVED: apply_stack_repair(ship)
    # The pre-sorted input guarantees Destination order (0 rehandles within stack)
    # and Weight order (Heavy at bottom) automatically.

    cost = CostReport(ship)

    lateral_instability = (
        abs(cost.row_moment) * 10.0 +
        abs(cost.tier_moment) * 1.0
    )

    bay_moment_penalty = cost.bay_moment ** 2

    return float(cost.rehandles), float(lateral_instability), float(bay_moment_penalty)


def evaluate_local(individual: Any, containers: List[Cont], num_bays: int, num_rows: int, max_tiers: int) -> Tuple[float, float, float]:
    """Evaluation function for non-pool contexts (uses passed parameters instead of globals)"""
    try:
        ind_as_list = cast(List[int], individual)

        ship = build_ship_from_individual(
            ind_as_list,
            containers,
            num_bays,
            num_rows,
            max_tiers
        )

        cost = CostReport(ship)

        # Ensure we have valid cost attributes
        rehandles = getattr(cost, 'rehandles', 0.0)
        row_moment = getattr(cost, 'row_moment', 0.0)
        tier_moment = getattr(cost, 'tier_moment', 0.0)
        bay_moment = getattr(cost, 'bay_moment', 0.0)

        lateral_instability = (
            abs(row_moment) * 10.0 +
            abs(tier_moment) * 1.0
        )

        bay_moment_penalty = bay_moment ** 2

        result = (float(rehandles), float(
            lateral_instability), float(bay_moment_penalty))

        # Validate result
        if len(result) != 3:
            raise ValueError(
                f"Expected 3 fitness values, got {len(result)}: {result}")

        return result

    except Exception as e:
        print(f"ERROR in evaluate_local: {e}")
        print(
            f"Cost object attributes: {dir(cost) if 'cost' in locals() else 'N/A'}")
        raise


def apply_memetic_local_search(individual: Any, containers: List[Cont], num_bays: int, num_rows: int, max_tiers: int) -> bool:
    """Bay-balance-aware local search with O(1) lookup."""
    ind_list = cast(List[int], individual)
    modified = False

    ship = build_ship_from_individual(
        ind_list, containers, num_bays, num_rows, max_tiers)

    cost = CostReport(ship)
    centre_bay = (num_bays - 1) / 2.0
    BAY_MOMENT_THRESHOLD = 0.5

    # --- Strategy 1: Fix Longitudinal Imbalance ---
    if abs(cost.bay_moment) > BAY_MOMENT_THRESHOLD:
        heavy_side = "fore" if cost.bay_moment < 0 else "aft"

        # Identify source/target bays
        if heavy_side == "fore":
            src_bays = range(0, int(centre_bay) + 1)
            tgt_bays = range(int(centre_bay) + 1, num_bays)
        else:
            src_bays = range(int(centre_bay) + 1, num_bays)
            tgt_bays = range(0, int(centre_bay) + 1)

        src_bays_set = set(src_bays)
        if not tgt_bays:
            return False

        # Find heaviest container on the heavy side
        best_container_idx = -1
        best_weight = -1.0

        for idx, c in enumerate(containers):
            # Fast check: look up current gene
            gene_b, _ = slots_from_column_index(ind_list[idx], num_rows)
            if gene_b in src_bays_set:
                w = c.get('weight', 0.0)
                if w > best_weight:
                    best_weight = w
                    best_container_idx = idx

        if best_container_idx != -1:
            new_b = random.choice(tgt_bays)
            new_r = random.randint(0, num_rows - 1)
            ind_list[best_container_idx] = index_from_column(
                new_b, new_r, num_rows)
            return True

    # --- Strategy 2: Fix Rehandles (Fallback) ---
    # Even with pre-sorting, rehandles can occur if we snake across bays badly?
    # Actually, with pre-sorting, rehandles are nearly impossible unless we run out of slots.
    # We keep this just in case logic drifts.

    blocking_containers = []
    for bay in ship:
        for row in bay:
            for t in range(1, len(row)):
                above = row[t]
                below = row[t - 1]
                # Pylance safety check
                if above and below and above['dest'] > below['dest']:
                    blocking_containers.append(above)

    if not blocking_containers:
        return False

    target_container = random.choice(blocking_containers)

    # OPTIMIZATION: O(1) Lookup
    # Build local ID map if global not available
    id_map = global_id_map if global_id_map else {
        c['id']: i for i, c in enumerate(containers)}

    if target_container['id'] in id_map:
        container_idx = id_map[target_container['id']]
        new_b = random.randint(0, num_bays - 1)
        new_r = random.randint(0, num_rows - 1)
        ind_list[container_idx] = index_from_column(new_b, new_r, num_rows)
        modified = True

    return modified


def apply_bay_balance_mutation(individual: Any, containers: List[Cont], num_bays: int, num_rows: int, max_tiers: int, indpb: float = 0.05) -> Tuple[Any]:
    ind_list = cast(List[int], individual)

    # Fast approximation of moments
    bay_weights = [0.0] * num_bays
    for idx, c in enumerate(containers):
        b, _ = slots_from_column_index(ind_list[idx], num_rows)
        bay_weights[b] += c.get('weight', 0.0)

    total_weight = sum(bay_weights) or 1.0
    centre_bay = (num_bays - 1) / 2.0
    current_moment = sum(bay_weights[b] * (b - centre_bay)
                         for b in range(num_bays)) / total_weight

    for idx in range(len(ind_list)):
        if random.random() < indpb:
            # Logic: move to lighter side
            if current_moment < 0:  # Fore heavy
                preferred_bays = range(num_bays // 2, num_bays)
            elif current_moment > 0:  # Aft heavy
                preferred_bays = range(0, num_bays // 2)
            else:
                preferred_bays = range(num_bays)

            if preferred_bays and random.random() < 0.7:
                new_b = random.choice(preferred_bays)
            else:
                new_b = random.randint(0, num_bays - 1)

            new_r = random.randint(0, num_rows - 1)
            ind_list[idx] = index_from_column(new_b, new_r, num_rows)

            # Simple incremental update could go here for speed,
            # but this is fast enough for mutation rate 0.05

    return (individual,)


def setup_toolbox(containers: List[Cont], num_bays: int, num_rows: int, max_tiers: int) -> base.Toolbox:
    num_containers = len(containers)

    # Clean up any existing creator classes to avoid conflicts
    if hasattr(creator, "FitnessMin"):
        del creator.FitnessMin
    if hasattr(creator, "Individual"):
        del creator.Individual

    # Create fresh fitness and individual classes
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
    creator.create("Individual", array.array, typecode='i',
                   fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    total_columns = num_bays * num_rows
    toolbox.register("attr_int", random.randint, 0, total_columns - 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_int, num_containers)  # type: ignore
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mutate", apply_bay_balance_mutation, containers=containers,
                     num_bays=num_bays, num_rows=num_rows, max_tiers=max_tiers, indpb=0.05)
    return toolbox


def solve_stowage_genetic(containers: List[Cont], num_bays: int, num_rows: int, max_tiers: int) -> Ship:
    # 1. GLOBAL SORT: Optimizes everything downstream
    # Sort by Discharge Port (DESC) -> Late ports at bottom
    # Then by Weight (DESC) -> Heavy at bottom (if ports equal)
    sorted_containers = sorted(
        containers,
        key=lambda k: (k['dest'], k['weight']),
        reverse=True
    )

    N = len(sorted_containers)
    raw_pop_size = min(200, max(50, 4 * N))
    pop_size = raw_pop_size - (raw_pop_size % 4)
    n_gen = min(300, max(50, 2 * N))

    toolbox = setup_toolbox(sorted_containers, num_bays, num_rows, max_tiers)
    toolbox.register("evaluate", evaluate_stowage_worker)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("select", tools.selNSGA2)

    cpu_count = multiprocessing.cpu_count()

    # Create population outside pool context
    pop: List[Any] = toolbox.population(n=pop_size)  # type: ignore

    # Smart seeds just need to snake-fill; sort order is already handled by the list
    num_seeds = max(2, int(pop_size * 0.25))

    # Simple snake generator since data is sorted
    # Generates indices [0, 0, 0, 1, 1, 1...] based on column fill
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

    # --- Initial evaluation using local function (no pool needed) ---
    fitnesses = [evaluate_local(
        ind, sorted_containers, num_bays, num_rows, max_tiers) for ind in pop]

    # DEBUG: Check what we're getting
    if fitnesses:
        print(f"DEBUG: First fitness value: {fitnesses[0]}")
        print(f"DEBUG: Type: {type(fitnesses[0])}")
        print(
            f"DEBUG: Length: {len(fitnesses[0]) if hasattr(fitnesses[0], '__len__') else 'N/A'}")

    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    tools.emo.assignCrowdingDist(pop)

    # Now start the pool for parallel evolution
    with multiprocessing.Pool(
        processes=cpu_count,
        initializer=init_worker,
        # Pass the SORTED containers to workers
        initargs=(sorted_containers, num_bays, num_rows, max_tiers)
    ) as pool:

        toolbox.register("map", pool.map)

        toolbox.register("map", pool.map)

        # --- Evolution Loop ---
        for gen in range(1, n_gen + 1):
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring]

            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.6:
                    toolbox.mate(ind1, ind2)  # type: ignore
                    del ind1.fitness.values
                    del ind2.fitness.values

            for ind in offspring:
                if random.random() < 0.4:
                    toolbox.mutate(ind)  # type: ignore
                    del ind.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            # Use pool.map for parallel evaluation during evolution
            fitnesses = toolbox.map(
                toolbox.evaluate, invalid_ind)  # type: ignore
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Local Search
            if gen % 10 == 0:
                offspring.sort(key=lambda x: sum(x.fitness.values))
                top_10 = int(len(offspring) * 0.10)
                # Note: pass sorted_containers here too
                for i in range(top_10):
                    elite = offspring[i]
                    mod = apply_memetic_local_search(
                        elite, sorted_containers, num_bays, num_rows, max_tiers)
                    if mod:
                        # Use local evaluation for single individual
                        new_fit = evaluate_local(
                            elite, sorted_containers, num_bays, num_rows, max_tiers)
                        elite.fitness.values = new_fit

            pop = toolbox.select(pop + offspring, pop_size)  # type: ignore

    # After pool is closed, do final selection
    pareto_fronts = tools.sortNondominated(
        pop, len(pop), first_front_only=True)
    best_ind = min(pareto_fronts[0],
                   key=lambda ind: sum(ind.fitness.values))

    best_ship = build_ship_from_individual(
        cast(List[int], best_ind),
        sorted_containers, num_bays, num_rows, max_tiers
    )
    return best_ship
