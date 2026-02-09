import argparse
import copy
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, TypeVar, Generic, Optional
from itertools import product

# --- 1. DATA MODELS (Unchanged) ---

Numeric = TypeVar("Numeric", int, float)


@dataclass
class Range(Generic[Numeric]):
    start: Numeric
    end: Numeric

    def __call__(self) -> Numeric:
        if isinstance(self.start, int) and isinstance(self.end, int):
            if self.start >= self.end:
                return self.start
            return random.randrange(self.start, self.end)
        return random.random() * (self.end - self.start) + self.start


@dataclass
class Container:
    id: int = field(init=False)
    weight: float
    dischargePort: int
    length: float
    reefer: bool
    _static_id: int = field(default=0, repr=False)

    def __post_init__(self):
        type(self)._static_id += 1
        self.id = type(self)._static_id

    def __repr__(self):
        r_tag = "R" if self.reefer else "S"
        return f"Cnt({self.id}, {self.length:.0f}ft, {r_tag}, W:{self.weight:.0f}, P:{self.dischargePort})"

    @classmethod
    def genRandom(cls, weight, port, lengths, thresh):
        return cls(weight(), port(), random.choice(lengths), random.random() > thresh)


@dataclass
class Slot:
    bay: int
    row: int
    tier: int
    max_weight: float
    length: float
    reefer: bool
    container: Optional[Container] = None
    @property
    def is_free(self): return self.container is None


@dataclass(frozen=True)
class SlotCoord:
    bay: int
    row: int
    tier: int


class Vessel:
    def __init__(self, bays, rows, tiers, max_w):
        self.bays, self.rows, self.tiers = bays, rows, tiers
        self.slots = {}
        for b, r, t in product(range(bays), range(rows), range(tiers)):
            plug = (r == rows - 1)
            l = 40.0 if (b % 2 == 0) else 20.0
            self.slots[SlotCoord(b, r, t)] = Slot(b, r, t, max_w, l, plug)

    def get_slot_at(self, coord): return self.slots.get(coord)
    def place(self, c, s): s.container = c
    @property
    def capacity(self): return len(self.slots)

    # Hard Constraints
    def check_hard_constraints(self, c, s):
        if not s.is_free:
            return False
        if c.weight > s.max_weight:
            return False
        if s.tier > 0:
            below = self.get_slot_at(SlotCoord(s.bay, s.row, s.tier - 1))
            if not below or below.is_free:
                return False
        if c.reefer and not s.reefer:
            return False
        if c.length != s.length:
            return False
        return True

# --- 2. HEURISTIC SCORING (The "Choice Function") ---
# Defined in Step 2.1 of the Action Plan


def score_move(vessel: Vessel, container: Container, slot: Slot) -> float:
    score = 0.0

    # Context: What is below us?
    slot_below = None
    if slot.tier > 0:
        slot_below = vessel.get_slot_at(
            SlotCoord(slot.bay, slot.row, slot.tier - 1))

    # A. Stability Bonus: Heavier containers low
    score -= (slot.tier * container.weight) / 1000.0

    # B. Balance: Center stowage preference
    center_row = (vessel.rows - 1) / 2.0
    dist_row = abs(slot.row - center_row)
    score -= dist_row * (container.weight / 1000.0)

    # C. Overstow Penalty
    if slot_below and slot_below.container:
        if container.dischargePort > slot_below.container.dischargePort:
            score -= 10000.0  # Massive penalty
        # D. Stacking Penalty (Anti-Crush)
        if container.weight > slot_below.container.weight:
            score -= 5000.0

    return score

# --- 3. RANDOMIZED GREEDY LOGIC (Phase 3) ---


def generate_one_plan(containers: List[Container], vessel: Vessel, alpha: float) -> Tuple[Vessel, List[Container]]:
    """
    Implements Step 3.1: The Construction Loop
    """
    # 0. Sort to guide the heuristic generally
    load_list = sorted(containers, key=lambda c: (
        c.dischargePort, c.weight), reverse=True)
    left_behind = []

    for container in load_list:
        # 1. Identify all valid (container, slot) pairs
        candidates = []
        for slot in vessel.slots.values():
            if vessel.check_hard_constraints(container, slot):
                s = score_move(vessel, container, slot)
                candidates.append((slot, s))

        if not candidates:
            left_behind.append(container)
            continue

        # 2. Score them & Sort
        candidates.sort(key=lambda x: x[1], reverse=True)

        # 3. Restricted Candidate List (RCL)
        # Instead of taking index 0 (Best), take top % (alpha)
        # "cutoff = int(len(scored_candidates) * alpha) + 1"
        cutoff = int(len(candidates) * alpha) + 1
        rcl = candidates[:cutoff]

        # 4. RANDOM SELECTION (The Monte Carlo element)
        selected_move = random.choice(rcl)
        target_slot = selected_move[0]

        # 5. Apply move
        vessel.place(container, target_slot)

        # Note: We don't remove from 'load_list' because we iterate over it directly

    return vessel, left_behind

# --- 4. MASTER MONTE CARLO LOOP (Step 3.2) ---


def calculate_total_cost(vessel: Vessel, leftovers: List[Container]) -> float:
    """Calculates the objective function value for comparison."""
    rehandles = 0
    tier_moment = 0.0

    for slot in vessel.slots.values():
        if slot.container:
            # Stability Cost
            tier_moment += slot.tier * slot.container.weight
            # Rehandle Cost
            if slot.tier > 0:
                below = vessel.get_slot_at(
                    SlotCoord(slot.bay, slot.row, slot.tier - 1))
                if below.container and slot.container.dischargePort > below.container.dischargePort:
                    rehandles += 1

    # Weighted Sum Objective
    cost = (rehandles * 1000.0) + (tier_moment * 0.1) + \
        (len(leftovers) * 5000.0)
    return cost


def monte_carlo_solver(args, initial_cargo, base_vessel):
    """
    Implements Step 3.2: The Master Monte Carlo Loop
    """
    best_cost = float('inf')
    best_vessel = None
    best_leftovers = []

    print(f"\n--- Starting Monte Carlo Simulation ---")
    print(f"Iterations: {args.iterations} | Alpha: {args.alpha}")

    for i in range(args.iterations):
        # Reset vessel and containers (Deep Copy)
        vessel_copy = copy.deepcopy(base_vessel)
        # Note: Container objects are immutable data, list copy is sufficient

        # Run Randomized Construction
        filled_vessel, leftovers = generate_one_plan(
            initial_cargo, vessel_copy, args.alpha)

        # Calculate Cost
        cost = calculate_total_cost(filled_vessel, leftovers)

        # Track Best
        if cost < best_cost:
            best_cost = cost
            best_vessel = filled_vessel
            best_leftovers = leftovers
            print(
                f"  > Iter {i+1}: New Best Cost {cost:.0f} (Left: {len(leftovers)})")

    return best_vessel, best_cost, best_leftovers

# --- 5. CLI RUNNER ---


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo Stowage Solver")
    parser.add_argument("--bays", nargs="+", type=int, default=[5])
    parser.add_argument("--rows", nargs="+", type=int, default=[5])
    parser.add_argument("--tiers", nargs="+", type=int, default=[5])
    parser.add_argument("--containers", nargs="+", type=int, default=[50])
    parser.add_argument("--weight", nargs="+", type=float,
                        default=[1000.0, 30000.0])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--alpha", type=float,
                        default=0.15, help="RCL parameter")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)

    def r(vals, f=False):
        if len(vals) == 1:
            return Range(vals[0], vals[0] if f else vals[0]+1)
        return Range(vals[0], vals[1])

    vessel = Vessel(r(args.bays)(), r(args.rows)(),
                    r(args.tiers)(), args.weight[-1])
    cargo = [Container.genRandom(r(args.weight, True), r(
        [1, 5]), [20, 40], 0.8) for _ in range(r(args.containers)())]

    print(
        f"Initialized. Ship: {vessel.capacity} slots. Cargo: {len(cargo)} items.")

    best_ves, cost, left = monte_carlo_solver(args, cargo, vessel)

    print("\n--- FINAL RESULT ---")
    print(f"Best Cost: {cost:.0f}")
    print(f"Left Behind: {len(left)}")

    # Stability Check
    tier_moments = sum(
        s.tier * s.container.weight for s in best_ves.slots.values() if s.container)
    print(f"Final Stability Score (Tier Moment): {tier_moments:.0f}")


if __name__ == "__main__":
    main()
