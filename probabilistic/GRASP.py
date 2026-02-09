from collections import defaultdict
from dataclasses import dataclass, field
from random import choice, randrange, random, seed
from typing import DefaultDict, Dict, List, Tuple, TypeVar, Generic, Optional
from itertools import product
import argparse
import copy  # Required for the simulation loop

# --- 1. DATA STRUCTURES & UTILS ---

Numeric = TypeVar("Numeric", int, float)


@dataclass
class Range(Generic[Numeric]):
    start: Numeric
    end: Numeric

    def __call__(self) -> Numeric:
        if isinstance(self.start, int) and isinstance(self.end, int):
            if self.start >= self.end:
                return self.start
            return randrange(self.start, self.end)
        else:
            return random() * (self.end - self.start) + self.start


type ChoiceDomain[T] = list[T]


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
    def genRandom(cls, weight: Range[float], port: Range[int], length_domain: ChoiceDomain[float], threshold: float):
        return cls(weight(), port(), choice(length_domain), random() > threshold)


@dataclass
class Slot:
    bay: int
    row: int
    tier: int
    max_weight: float
    length: float
    reefer: bool
    container: Container | None = None

    @property
    def is_free(self):
        return self.container is None


@dataclass(frozen=True)
class SlotCoord:
    bay: int
    row: int
    tier: int


class Vessel:
    def __init__(self, bays: int, rows: int, tiers: int, weight: Range[float]):
        self.bays = bays
        self.rows = rows
        self.tiers = tiers
        self.slots: Dict[SlotCoord, Slot] = {}

        for b, r, t in product(range(self.bays), range(self.rows), range(self.tiers)):
            # Deterministic Configuration:
            # 1. Last Row = Reefer Plug
            has_plug = (r == self.rows - 1)
            # 2. Even Bays = 40ft, Odd Bays = 20ft
            slot_len = 40.0 if (b % 2 == 0) else 20.0

            self.slots[SlotCoord(b, r, t)] = Slot(
                b, r, t, weight.end, slot_len, has_plug
            )

    @property
    def capacity(self):
        return len(self.slots)

    def get_slot_at(self, coord: SlotCoord) -> Slot | None:
        return self.slots.get(coord)

    def place(self, container: Container, slot: Slot) -> None:
        slot.container = container

    def check_hard_constraints(self, container: Container, slot: Slot) -> bool:
        # 1. Occupancy
        if not slot.is_free:
            return False
        # 2. Structural Limit
        if container.weight > slot.max_weight:
            return False
        # 3. Physics (Gravity)
        if slot.tier > 0:
            slot_below = self.get_slot_at(
                SlotCoord(slot.bay, slot.row, slot.tier - 1))
            if slot_below is None or slot_below.is_free:
                return False
        # 4. Compatibility
        if container.reefer and not slot.reefer:
            return False
        if container.length != slot.length:
            return False
        return True

# --- 2. PHYSICS ENGINE ---


class PhysicsUtils:
    @staticmethod
    def calculate_rehandles(vessel: Vessel) -> int:
        total = 0
        for b, r in product(range(vessel.bays), range(vessel.rows)):
            stack = [vessel.get_slot_at(SlotCoord(b, r, t))
                     for t in range(vessel.tiers)]
            filled = [s for s in stack if s and s.container]
            for i in range(1, len(filled)):
                below = filled[i-1].container
                above = filled[i].container
                if above.dischargePort > below.dischargePort:
                    total += 1
        return total

    @staticmethod
    def calculate_moments(vessel: Vessel) -> dict:
        total_weight = 0.0
        moment_bay = 0.0
        moment_row = 0.0
        moment_tier = 0.0

        center_bay = (vessel.bays - 1) / 2.0
        center_row = (vessel.rows - 1) / 2.0

        for slot in vessel.slots.values():
            if slot.container:
                w = slot.container.weight
                total_weight += w
                moment_bay += w * (slot.bay - center_bay)
                moment_row += w * (slot.row - center_row)
                moment_tier += w * slot.tier

        if total_weight == 0:
            return {"bay": 0.0, "row": 0.0, "tier": 0.0}

        return {
            "bay": moment_bay / total_weight,
            "row": moment_row / total_weight,
            "tier": moment_tier / total_weight
        }

# --- 3. HEURISTIC SCORING ---


def score_move(vessel: Vessel, container: Container, slot: Slot) -> float:
    score = 0.0
    slot_below = None
    if slot.tier > 0:
        slot_below = vessel.get_slot_at(
            SlotCoord(slot.bay, slot.row, slot.tier - 1))

    # A: Stability (Lower tiers prefer heavy weights)
    score -= (slot.tier * container.weight) / 1000.0

    # B: Balance (Center rows prefer heavy weights)
    center_row = (vessel.rows - 1) / 2.0
    dist_row = abs(slot.row - center_row)
    score -= dist_row * (container.weight / 1000.0)

    # C: Overstowage Avoidance
    if slot_below and slot_below.container:
        if container.dischargePort > slot_below.container.dischargePort:
            score -= 10000.0
        if container.weight > slot_below.container.weight:
            score -= 5000.0

    return score

# --- 4. PROBABILISTIC SOLVER (GRASP Construction) ---


def randomized_greedy_solver(containers: List[Container], vessel: Vessel, alpha: float) -> Tuple[List, List]:
    """
    Constructs a solution using a Restricted Candidate List (RCL).
    alpha: 0.0 = Pure Greedy (Deterministic Best-Fit).
           1.0 = Pure Random (Any valid slot).
    """
    # Still sort to give general guidance, but randomness will handle the details
    load_list = sorted(containers, key=lambda c: (
        c.dischargePort, c.weight), reverse=True)

    plan = []
    left_behind = []

    for container in load_list:
        valid_moves = []

        # 1. Identify all valid slots and score them
        for slot in vessel.slots.values():
            if vessel.check_hard_constraints(container, slot):
                s = score_move(vessel, container, slot)
                valid_moves.append((slot, s))

        if not valid_moves:
            left_behind.append(container)
            continue

        # 2. Build the RCL (Restricted Candidate List)
        # Sort best scores to the top
        valid_moves.sort(key=lambda x: x[1], reverse=True)

        # Calculate how many candidates to consider based on Alpha
        # Example: 10 valid slots, Alpha 0.2 => Consider Top 2 slots
        limit = max(1, int(len(valid_moves) * alpha))

        # Slice the list to get top candidates
        rcl = valid_moves[:limit]

        # 3. Probabilistic Selection
        # Pick one random slot from the top tier
        selected_move = choice(rcl)
        best_slot = selected_move[0]

        # Apply
        vessel.place(container, best_slot)
        plan.append((container, best_slot))

    return plan, left_behind

# --- 5. SIMULATION LOOP ---


@dataclass
class SolutionReport:
    cost: float
    rehandles: int
    unstowed: int
    moments: dict


def solve_probabilistic(args, initial_cargo, base_vessel):
    best_cost = float('inf')
    best_report = None
    best_vessel = None

    print(f"\n--- Probabilistic Simulation (GRASP) ---")
    print(f"Iterations: {args.iterations} | Alpha: {args.alpha}")

    for i in range(args.iterations):
        # A. Copy State (Crucial for simulation)
        current_vessel = copy.deepcopy(base_vessel)  # Requires valid copy
        # Shallow copy of list is sufficient for read-only containers
        current_cargo = list(initial_cargo)

        # B. Run Randomized Construction
        _, left_behind = randomized_greedy_solver(
            current_cargo, current_vessel, args.alpha)

        # C. Evaluate This Run
        rehandles = PhysicsUtils.calculate_rehandles(current_vessel)
        moments = PhysicsUtils.calculate_moments(current_vessel)

        # Cost Function (Weights: Overstow=1000, Stability=100, Unstowed=10000)
        # We weigh Unstowed heavily because "leaving cargo behind" is the worst outcome.
        run_cost = (rehandles * 1000.0) + \
                   (abs(moments['row']) * 100.0) + \
                   (moments['tier'] * 10.0) + \
                   (len(left_behind) * 10000.0)

        # D. Track Best
        if run_cost < best_cost:
            best_cost = run_cost
            best_vessel = current_vessel
            best_report = SolutionReport(
                run_cost, rehandles, len(left_behind), moments)
            print(
                f"  > Iter {i+1}: New Best Cost {best_cost:.1f} (Unstowed: {len(left_behind)})")

    return best_vessel, best_report

# --- 6. CLI ---


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probabilistic Stowage Solver (GRASP)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    ship_group = parser.add_argument_group("Ship Configuration")
    ship_group.add_argument("--bays", nargs="+", type=int, default=[5])
    ship_group.add_argument("--rows", nargs="+", type=int, default=[5])
    ship_group.add_argument("--tiers", nargs="+", type=int, default=[5])

    cargo_group = parser.add_argument_group("Cargo Configuration")
    cargo_group.add_argument("--containers", nargs="+",
                             type=int, default=[50, 100])
    cargo_group.add_argument("--weight", nargs="+",
                             type=float, default=[1000.0, 30000.0])
    cargo_group.add_argument("--ports", nargs="+", type=int, default=[1, 5])

    prob_group = parser.add_argument_group("Probabilistic Parameters")
    prob_group.add_argument("--iterations", type=int, default=100,
                            help="Number of Monte Carlo simulations to run.")
    prob_group.add_argument("--alpha", type=float, default=0.2,
                            help="Randomness factor (0.0=Greedy, 1.0=Random).")
    prob_group.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    def to_range(values: List[Numeric], is_float: bool = False) -> Range:
        if len(values) == 1:
            start = values[0]
            if is_float:
                return Range(float(start), float(start))
            else:
                return Range(int(start), int(start) + 1)
        elif len(values) == 2:
            return Range(values[0], values[1])
        else:
            raise ValueError("Arguments must be 1 or 2 values")

    if args.seed is not None:
        seed(args.seed)
        print(f"--- SEED: {args.seed} ---")

    try:
        # 1. Setup
        vessel = Vessel(
            to_range(args.bays)(),
            to_range(args.rows)(),
            to_range(args.tiers)(),
            to_range(args.weight, is_float=True)
        )

        gen_amt = to_range(args.containers)()
        # Use simple fixed thresholds for CLI demo
        cargo = [Container.genRandom(to_range(args.weight, True), to_range(args.ports), [20, 40], 0.8)
                 for _ in range(gen_amt)]

        print(
            f"\n[GENERATION] Slots: {vessel.capacity} | Containers: {gen_amt}")

        # 2. Run Simulation
        best_ves, report = solve_probabilistic(args, cargo, vessel)

        if not report or not best_ves:
            return

        # 3. Final Output
        print(f"\n--- FINAL BEST RESULT ---")
        print(f"Cost: {report.cost:.1f}")
        print(f"Containers Left Behind: {report.unstowed}")
        print(f"Rehandles: {report.rehandles}")
        print(
            f"Vertical Stability (Tier Moment): {report.moments['tier']:.2f}")
        print(f"Transverse Balance (Row Moment): {report.moments['row']:.2f}")

    except ValueError as e:
        print(f"Error parsing inputs: {e}")


if __name__ == "__main__":
    main()
