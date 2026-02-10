# genetic.py
# Multi-objective GA for container stowage using DEAP (NSGA-II)
# Objectives: minimize (rehandles, bay_moment, row_moment, tier_moment)
# Reference: Container Vessel Stowage Planning System using Genetic Algorithm

from common import RehandlesNumber, BayMoment, RowMoment, TierMoment
from common import Cont, CostReport, Ship
from deap import base, creator, tools, algorithms
from typing import List, Dict, Tuple, Optional
import math
from typing import List, Tuple
import random
import array
<< << << < HEAD
== == == =
>>>>>> > 45ff7aab43b2a5dea7737bb2753cead4d59a9918

# DEAP

# Import metric calculators from your common.py
# Ensure common.py has these functions: RehandlesNumber, BayMoment, RowMoment, TierMoment
<< << << < HEAD
== == == =
>>>>>> > 45ff7aab43b2a5dea7737bb2753cead4d59a9918

# ------------------------------------------------------------------------------
# 1. Helper Functions
# ------------------------------------------------------------------------------


def slots_from_index(idx: int, num_bays: int, num_rows: int, max_tiers: int) -> Tuple[int, int, int]:
    """Decodes a linear index back to (bay, row, tier)."""
    per_bay = num_rows * max_tiers
    b = idx // per_bay
    rem = idx % per_bay
    r = rem // max_tiers
    t = rem % max_tiers
    return b, r, t


def build_ship_from_individual(individual: List[int], containers: List[Cont], num_bays: int, num_rows: int, max_tiers: int):
    """
    Constructs the 3D ship matrix from the genetic individual.
    Strategy: Attempt to place container at 'individual[i]' slot. 
    If occupied/invalid, search linearly for the next open slot to ensure validity.
    """
    # Initialize empty ship: Bay -> Row -> Tier
    ship: Ship = [[[None for _ in range(max_tiers)]
                   for _ in range(num_rows)] for _ in range(num_bays)]

    # Track occupied slots to avoid overwriting
    # Linear size of the ship
    total_slots = num_bays * num_rows * max_tiers
    occupied = [False] * total_slots

    # Determine order: The individual defines the PREFERRED slot for each container
    for i, container in enumerate(containers):
        preferred_idx = individual[i]

        placed = False
        # Try preferred slot first, then look for nearest empty
        # We search modulo total_slots to wrap around if needed
        for offset in range(total_slots):
            current_idx = (preferred_idx + offset) % total_slots

            # Check if strictly valid coordinates
            b, r, t = slots_from_index(
                current_idx, num_bays, num_rows, max_tiers)

            # Skip invalid indices (sanity check, though math ensures they are in bound)
            if b >= num_bays or r >= num_rows or t >= max_tiers:
                continue

            # Check logical validity (GRAVITY):
            # A container can only be placed at tier T if T=0 OR T-1 is occupied.
            # This enforces "floating container" physical constraint.
            if t > 0:
                if ship[b][r][t-1] is None:
                    continue  # Cannot float in air

            if ship[b][r][t] is None:
                ship[b][r][t] = container
                placed = True
                break

        if not placed:
            # Fallback: if ship is 100% full or no valid gravity spot found (rare)
            # We append to a 'virtual' overflow list or just ignore (penalize later)
            pass

    return ship

# ------------------------------------------------------------------------------
# 2. GA Setup (NSGA-II)
# ------------------------------------------------------------------------------


def setup_toolbox(num_containers: int, num_bays: int, num_rows: int, max_tiers: int):
    # Minimization problem for 4 objectives
    # Weights are negative because we want to MINIMIZE all values
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness,
                       weights=(-1.0, -1.0, -1.0, -1.0))

    if not hasattr(creator, "Individual"):
        creator.create("Individual", array.array, typecode='i',
                       fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # Attribute generator: Random integer representing a slot index
    total_slots = num_bays * num_rows * max_tiers
    toolbox.register("attr_int", random.randint, 0, total_slots - 1)

    # Structure initializers
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_int, num_containers)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    return toolbox

# ------------------------------------------------------------------------------
# 3. Evaluation Function
# ------------------------------------------------------------------------------


def evaluate_stowage(individual, containers, num_bays, num_rows, max_tiers):
    """
    Evaluates the individual based on the 4 constraints/objectives.
    """
    # 1. Decode individual into a Ship structure
    ship = build_ship_from_individual(
        individual, containers, num_bays, num_rows, max_tiers)

    # 2. Calculate Metrics using common.py functions
    # Note: We must handle cases where common functions expect specific formats
    cost = CostReport(ship)

    rehandles = cost.rehandles

    # Calculate moments.
    # Note: We divide by size to normalize, or just return raw moment.
    # We use absolute value of moment because usually we want CoG to be at Center (0.0).
    # If the helper returns the actual CoG position, we want abs(CoG - Center).
    # Assuming common.py returns the CoG value directly:

    b_mom = abs(cost.bay_moment)
    r_mom = abs(cost.row_moment)
    # We usually want Min Tier Moment (lower CoG), so no Abs() if it is just 'Sum(Weight*Height)'
    t_mom = abs(cost.tier_moment)

    # For Tier Moment, strictly speaking, lower is better for stability.
    # If TierMoment returns a "center" (like 2.5), we want to minimize it.

    return float(rehandles), float(b_mom), float(r_mom), float(t_mom)

# ------------------------------------------------------------------------------
# 4. Main Solving Function
# ------------------------------------------------------------------------------


def solve_stowage_genetic(containers: List[Cont], num_bays: int, num_rows: int, max_tiers: int,
                          pop_size=100, n_gen=50):

    toolbox = setup_toolbox(len(containers), num_bays, num_rows, max_tiers)

    # Register Operators
    toolbox.register("evaluate", evaluate_stowage, containers=containers,
                     num_bays=num_bays, num_rows=num_rows, max_tiers=max_tiers)

    # Crossover: TwoPoint is good for integer sequences
    toolbox.register("mate", tools.cxTwoPoint)

    # Mutation: Shuffle indices (reordering) or Uniform Int (changing slot target)
    # We combine both: probability of changing a target slot vs swapping containers
    toolbox.register("mutate", tools.mutUniformInt, low=0,
                     up=(num_bays*num_rows*max_tiers)-1, indpb=0.05)

    # Selection: NSGA-II is standard for Multi-Objective
    toolbox.register("select", tools.selNSGA2)

    # Initialize Population
    pop = toolbox.population(n=pop_size)

    # Statistics to track progress
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", lambda x: tuple(
        map(min, zip(*x))))  # Min of each objective
    stats.register("avg", lambda x: tuple(
        map(lambda y: sum(y)/len(y), zip(*x))))

    # Run Algorithm
    # mu: population size, lambda_: offspring size (usually same as mu)
    # cxpb: crossover prob, mutpb: mutation prob
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=pop_size, lambda_=pop_size,
                                             cxpb=0.7, mutpb=0.2, ngen=n_gen,
                                             stats=stats, verbose=False)

    # Extract Pareto Front (Best tradeoffs)
    pareto_front = tools.sortNondominated(
        pop, len(pop), first_front_only=True)[0]

    # Select a single "best" solution for the return value
    # Strategy: Prioritize Rehandles (index 0), then sum of moments
    best_ind = min(pareto_front, key=lambda ind: (
        ind.fitness.values[0], sum(ind.fitness.values[1:])))

    best_ship = build_ship_from_individual(
        best_ind, containers, num_bays, num_rows, max_tiers)

    return CostReport(best_ship)
