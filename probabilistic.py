import argparse
import copy
import random
<<<<<<< HEAD
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, TypeVar, Generic, Optional
from itertools import product

# --- DATA STRUCTURES ---

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
        else:
            return random.random() * (self.end - self.start) + self.start


@dataclass
class Container:
    id: int = field(init=False)
    weight: float
    dischargePort: int
    is_reefer: bool = False
    length: int = 20
    _static_id: int = field(default=0, repr=False)

    def __post_init__(self):
        type(self)._static_id += 1
        self.id = type(self)._static_id

    def __repr__(self):
        r_tag = "R" if self.is_reefer else "S"
        return f"Cnt({self.id}, {self.length}ft, {r_tag}, W:{self.weight:.0f}, P:{self.dischargePort})"

    @classmethod
    def genRandom(cls, weight: Range[float], port: Range[int]):
        is_r = random.random() < 0.10
        l = 40 if random.random() < 0.30 else 20
        return cls(weight(), port(), is_reefer=is_r, length=l)


@dataclass
class Slot:
    bay: int
    row: int
    tier: int
    max_weight: float
    is_reefer: bool = False
    length: int = 20
    container: Optional[Container] = None

    @property
    def is_free(self):
        return self.container is None


@dataclass(frozen=True)
class SlotCoord:
    bay: int
    row: int
    tier: int


@dataclass
class Vessel:
    bays: int
    rows: int
    tiers: int
    max_weight: float
    slots: Dict[SlotCoord, Slot] = field(
        default_factory=dict, init=False, repr=False)

    @property
    def capacity(self):
        return len(self.slots)

    def __post_init__(self):
        for b, r, t in product(range(self.bays), range(self.rows), range(self.tiers)):
            # Configuration Logic
            has_plug = (r == self.rows - 1)
            slot_len = 40 if (b % 2 == 0) else 20
            self.slots[SlotCoord(b, r, t)] = Slot(
                b, r, t, self.max_weight, is_reefer=has_plug, length=slot_len
            )

    def get_slot_at(self, coord: SlotCoord) -> Optional[Slot]:
        return self.slots.get(coord)

    def place(self, container: Container, slot: Slot):
        slot.container = container

    def check_hard_constraints(self, container: Container, slot: Slot) -> bool:
        if not slot.is_free:
            return False
        if container.weight > slot.max_weight:
            return False

        # Floating check
        if slot.tier > 0:
            below = self.get_slot_at(
                SlotCoord(slot.bay, slot.row, slot.tier - 1))
            if below is None or below.is_free:
                return False

        # Type/Size checks
        if container.is_reefer and not slot.is_reefer:
            return False
        if container.length != slot.length:
            return False
        return True

# --- NEW: PHYSICS ENGINE (Ported from Snippet) ---


class PhysicsUtils:
    @staticmethod
    def calculate_rehandles(vessel: Vessel) -> int:
        total = 0
        for b, r in product(range(vessel.bays), range(vessel.rows)):
            stack_slots = [vessel.get_slot_at(
                SlotCoord(b, r, t)) for t in range(vessel.tiers)]
            filled_slots = [s for s in stack_slots if s and s.container]

            for i in range(1, len(filled_slots)):
                below = filled_slots[i-1].container
                above = filled_slots[i].container
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

# --- UPDATED HEURISTIC SCORING ---
=======
from typing import List, Tuple

from common import Vessel, Container, Slot, SlotCoord, calculate_cost, Range

# ==========================================
# 1. HEURISTIC ENGINE (Used for Expansion & Rollout)
# ==========================================
>>>>>>> a1dc743 ([Feat]: common interface common.py)


def score_move(vessel: Vessel, container: Container, slot: Slot) -> float:
    score = 0.0
    slot_below = None
    if slot.tier > 0:
        slot_below = vessel.get_slot_at(
            SlotCoord(slot.bay, slot.row, slot.tier - 1))

    # 1. Stability (Minimize Tier Moment)
    # Penalize high placements, scaled by weight
    score -= (slot.tier * container.weight) / 1000.0

    # 2. Balance (Minimize Bay/Row Moments)
    # Prefer placing heavy items near the geometric center
    center_bay = (vessel.bays - 1) / 2.0
    center_row = (vessel.rows - 1) / 2.0
    dist_bay = abs(slot.bay - center_bay)
    dist_row = abs(slot.row - center_row)

    score -= (dist_bay + dist_row) * (container.weight / 1000.0)

    # 3. Rehandles (Hard Penalty)
    if slot_below and slot_below.container:
        if container.dischargePort > slot_below.container.dischargePort:
            score -= 10000.0
        # 4. Anti-Crush (Auxiliary)
        if container.weight > slot_below.container.weight:
            score -= 5000.0

    return score

<<<<<<< HEAD
# --- SOLVER ENGINE ---
=======

# ==========================================
# 3. MONTE CARLO TREE SEARCH (MCTS)
# ==========================================
>>>>>>> a1dc743 ([Feat]: common interface common.py)


def randomized_greedy_solver(containers: List[Container], vessel: Vessel, alpha: float) -> Tuple[List, List]:
    # Sort Phase
    load_list = sorted(containers, key=lambda c: (
        c.dischargePort, c.weight), reverse=True)
    plan = []
    left_behind = []

    for container in load_list:
        valid_moves = []
        for slot in vessel.slots.values():
            if vessel.check_hard_constraints(container, slot):
                s = score_move(vessel, container, slot)
                valid_moves.append((slot, s))

        if not valid_moves:
            left_behind.append(container)
            continue

        valid_moves.sort(key=lambda x: x[1], reverse=True)
        cutoff = int(len(valid_moves) * alpha) + 1
        rcl = valid_moves[:cutoff]
        target_slot = random.choice(rcl)[0]

        vessel.place(container, target_slot)
        plan.append((container, target_slot))

    return plan, left_behind


@dataclass
class PenaltyWeights:
    overstow: float
    stability: float  # Controls Beta (CoG) influence


def run_monte_carlo(args, initial_cargo, base_vessel, penalties):
    best_cost = float('inf')
    best_metrics = {}
    best_vessel = None

    print(f"\n--- Starting Monte Carlo ({args.iterations} iters) ---")

    for i in range(args.iterations):
        current_vessel = copy.deepcopy(base_vessel)
        plan, leftovers = randomized_greedy_solver(
            initial_cargo, current_vessel, alpha=args.alpha)

        # METRIC CALCULATION (Matching MILP)
        rehandles = PhysicsUtils.calculate_rehandles(current_vessel)
        moments = PhysicsUtils.calculate_moments(current_vessel)

        # COST FUNCTION: Z = Alpha*Rehandles + Beta*Moments
        # We use penalties.overstow for Alpha, penalties.stability for Beta
        cost = (rehandles * penalties.overstow) + \
               (abs(moments['bay']) * penalties.stability) + \
               (abs(moments['row']) * penalties.stability) + \
               (moments['tier'] * penalties.stability)

        cost += len(leftovers) * 5000.0  # Operational penalty

        if cost < best_cost:
            best_cost = cost
            best_vessel = current_vessel
            best_metrics = {"rehandles": rehandles,
                            "moments": moments, "left": len(leftovers)}
            print(
                f"  [Iter {i+1}] New Best Z: {cost:.2f} (Rehandles: {rehandles}, TierM: {moments['tier']:.2f})")

    return best_metrics, best_vessel, best_cost

# --- MAIN CLI ---


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bays", nargs="+", type=int, default=[5])
    parser.add_argument("--rows", nargs="+", type=int, default=[5])
    parser.add_argument("--tiers", nargs="+", type=int, default=[5])
    parser.add_argument("--containers", nargs="+", type=int, default=[50])
    parser.add_argument("--weight", nargs="+", type=float,
                        default=[1000.0, 30000.0])
    parser.add_argument("--ports", nargs="+", type=int, default=[1, 5])
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)

    def to_range(vals, is_float=False):
        if len(vals) == 1:
            return Range(vals[0], vals[0] if is_float else vals[0]+1)
        return Range(vals[0], vals[1])

    vessel = Vessel(to_range(args.bays)(), to_range(args.rows)
                    (), to_range(args.tiers)(), args.weight[-1])
    gen_amt = to_range(args.containers)()
    cargo = [Container.genRandom(to_range(args.weight, True), to_range(
        args.ports)) for _ in range(gen_amt)]

    # Weights: Overstow is expensive (ALPHA), Stability is secondary (BETA)
    penalties = PenaltyWeights(overstow=1000.0, stability=10.0)

    metrics, best_ves, cost = run_monte_carlo(args, cargo, vessel, penalties)

    print(f"\n--- FINAL RESULTS ---")
    print(f"Objective Value (Z): {cost:.2f}")
    print(f"Total Rehandles: {metrics['rehandles']}")
    print(
        f"CoG Deviations: Bay={metrics['moments']['bay']:.2f}, Row={metrics['moments']['row']:.2f}, Tier={metrics['moments']['tier']:.2f}")
    print(f"Unstowed Containers: {metrics['left']}")

    # Visual check of Balance
    print("\n--- BALANCE CHECK (Center Row) ---")
    mid_row = (vessel.rows - 1) // 2
    for b in range(vessel.bays):
        filled_weight = 0
        for t in range(vessel.tiers):
            s = best_ves.get_slot_at(SlotCoord(b, mid_row, t))
            if s and s.container:
                filled_weight += s.container.weight
        print(f"Bay {b} Row {mid_row} (Center): {filled_weight:.0f} kg")
