from __future__ import annotations
from typing import List, Optional, Tuple
from itertools import product
from typing import Dict, List, Tuple, TypeVar, Generic, Optional
from dataclasses import dataclass, field

import argparse
import copy
import math
import random
from typing import List, Tuple

from common import Vessel, Container, Slot, SlotCoord, calculate_cost, Range

# ==========================================
# 1. HEURISTIC ENGINE (Used for Expansion & Rollout)
# ==========================================
<< << << < HEAD
<< << << < HEAD

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
== == == =
== == == =
>>>>>> > c9faeeb([Refac]: pylance strict)


# ==========================================
# 1. HEURISTIC ENGINE (Used for Expansion & Rollout)
# ==========================================
>>>>>> > a1dc743([Feat]: common interface common.py)


def score_move(vessel: Vessel, container: Container, slot: Slot) -> float:
    score = 0.0
    slot_below = None
    if slot.tier > 0:


<< << << < HEAD
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

=======
        under = vessel.get_slot_at(
            SlotCoord(slot.bay, slot.row, slot.tier - 1))
        if under:
            below = under.container
            if below:
                if container.dischargePort > below.dischargePort:
                    score -= 10000.0
                if container.weight > below.weight:
                    score -= 5000.0
>>>>>>> c9faeeb ([Refac]: pylance strict)
    return score

<<<<<<< HEAD
# --- SOLVER ENGINE ---
=======

# ==========================================
# 3. MONTE CARLO TREE SEARCH (MCTS)
# ==========================================
>>>>>>> a1dc743 ([Feat]: common interface common.py)


<<<<<<< HEAD
def randomized_greedy_solver(containers: List[Container], vessel: Vessel, alpha: float) -> Tuple[List, List]:
    # Sort Phase
    load_list = sorted(containers, key=lambda c: (
=======
class MCTSNode:
    def __init__(self, vessel_state: Vessel, remaining_cargo: List[Container], parent: Optional[MCTSNode] = None):
        self.vessel: Vessel = vessel_state
        self.cargo: List[Container] = remaining_cargo
        self.parent: Optional[MCTSNode] = parent
        self.children: List[MCTSNode] = []
        self.visits: int = 0
        self.value: float = 0.0

        self.untried_moves: Optional[List[Tuple[Container, Slot]]] = None

    @property
    def is_terminal(self):
        return len(self.cargo) == 0

    @property
    def is_fully_expanded(self):
        return self.untried_moves is not None and len(self.untried_moves) == 0

    def get_legal_moves(self) -> List[Tuple[Container, Slot]]:
        """Identify the next container and top valid slots."""
        if not self.cargo:
            return []

        # Strategy: Strict Ordering. We only try to place the NEXT container.
        next_c = self.cargo[0]

        candidates: List[Tuple[Slot, float]] = []
        for s in self.vessel.slots.values():
            if self.vessel.check_hard_constraints(next_c, s):
                candidates.append((s, score_move(self.vessel, next_c, s)))

        # Heuristic Pruning: Only consider top 5 slots to keep tree manageable
        # This is critical for performance in Python
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_k = candidates[:5]

        return [(next_c, x[0]) for x in top_k]


def mcts_search(root_vessel: Vessel, initial_cargo: List[Container], iterations: int = 1000):
    # Sort cargo once (Global Ordering Strategy) [cite: 151]
    sorted_cargo = sorted(initial_cargo, key=lambda c: (
>>>>>>> c9faeeb ([Refac]: pylance strict)
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

<<<<<<< HEAD
    return plan, left_behind
=======
        # 1. SELECTION (Traverse down to a leaf)
        # Use UCB1: node_val/visits + C * sqrt(ln(parent_visits)/visits)
        while not node.is_terminal and node.is_fully_expanded:
            def keyFunc(c: MCTSNode) -> float:
                if not node:
                    raise ValueError("node is None!")
                if c.visits == 0:
                    return float('inf')

                return (c.value / c.visits) + 1.41 * math.sqrt(math.log(node.visits) / c.visits)
            node = max(node.children, key=keyFunc)
>>>>>>> c9faeeb ([Refac]: pylance strict)


@dataclass
class PenaltyWeights:
    overstow: float
    stability: float  # Controls Beta (CoG) influence

<<<<<<< HEAD

def run_monte_carlo(args, initial_cargo, base_vessel, penalties):
    best_cost = float('inf')
    best_metrics = {}
    best_vessel = None

    print(f"\n--- Starting Monte Carlo ({args.iterations} iters) ---")

    for i in range(args.iterations):
        current_vessel = copy.deepcopy(base_vessel)
        plan, leftovers = randomized_greedy_solver(
            initial_cargo, current_vessel, alpha=args.alpha)
=======
                # Clone state for new node
                new_vessel = copy.deepcopy(node.vessel)
                s = new_vessel.get_slot_at(
                    SlotCoord(slot.bay, slot.row, slot.tier))
                if s:
                    new_vessel.place(container, s)
                    new_cargo = node.cargo[1:]

                    child_node = MCTSNode(new_vessel, new_cargo, parent=node)
                    node.children.append(child_node)
                    node = child_node

        # 3. SIMULATION (Rollout)
        # Use Randomized Greedy logic to finish the plan from this node
        sim_vessel = copy.deepcopy(node.vessel)
        sim_cargo = list(node.cargo)
        sim_leftovers: List[Container] = []

        # Fast Greedy Rollout
        for c in sim_cargo:
            candidates: List[Tuple[Slot, float]] = []
            for s in sim_vessel.slots.values():
                if sim_vessel.check_hard_constraints(c, s):
                    candidates.append((s, score_move(sim_vessel, c, s)))
>>>>>>> c9faeeb ([Refac]: pylance strict)

        # METRIC CALCULATION (Matching MILP)
        rehandles = PhysicsUtils.calculate_rehandles(current_vessel)
        moments = PhysicsUtils.calculate_moments(current_vessel)

            if candidates:
                # Greedy choice (Top 1) for speed in rollout
                candidates.sort(key=lambda x: x[1], reverse=True)
                target = candidates[0][0]
                sim_vessel.place(c, target)
            else:
                sim_leftovers.append(c)

        # 4. BACKPROPAGATION
        # Convert Cost to Reward (Lower cost = Higher Reward)
        # Using simple normalization 100000 / cost
        cost = calculate_cost(sim_vessel, sim_leftovers)
        reward = 1.0 / (1.0 + cost)

        # Update Best Found
        if cost < min_global_cost:
            min_global_cost = cost
            best_global_plan = sim_vessel
            print(
                f"  [Iter {i+1}] New Best Z: {cost:.2f} (Rehandles: {rehandles}, TierM: {moments['tier']:.2f})")

<<<<<<< HEAD
    return best_metrics, best_vessel, best_cost
=======
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
>>>>>>> c9faeeb ([Refac]: pylance strict)

# ==========================================
# 4. CLI RUNNER
# ==========================================


def main():
    parser = argparse.ArgumentParser(description="MCTS Stowage Solver")
    parser.add_argument("--bays", nargs="+", type=int, default=[5])
    parser.add_argument("--rows", nargs="+", type=int, default=[5])
    parser.add_argument("--tiers", nargs="+", type=int, default=[5])
    parser.add_argument("--containers", nargs="+", type=int, default=[50])
    parser.add_argument("--weight", nargs="+", type=float,
                        default=[1000.0, 30000.0])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

<<<<<<< HEAD
    def to_range(vals, is_float=False):
=======
    def ri(vals: List[int], f: bool = False):
>>>>>>> c9faeeb ([Refac]: pylance strict)
        if len(vals) == 1:
            return Range(vals[0], vals[0] if f else vals[0]+1)
        return Range(vals[0], vals[1])

<<<<<<< HEAD
    vessel = Vessel(to_range(args.bays)(), to_range(args.rows)
                    (), to_range(args.tiers)(), args.weight[-1])
    gen_amt = to_range(args.containers)()
    cargo = [Container.genRandom(to_range(args.weight, True), to_range(
        args.ports)) for _ in range(gen_amt)]
=======
    def rf(vals: List[float], f: bool = False):
        if len(vals) == 1:
            return Range(vals[0], vals[0] if f else vals[0]+1)
        return Range(vals[0], vals[1])

    vessel = Vessel(ri(args.bays)(), ri(args.rows)(),
                    ri(args.tiers)())
    cargo = [Container.gen_random(rf(args.weight, True), ri(
        [1, 5])) for _ in range(ri(args.containers)())]
>>>>>>> c9faeeb ([Refac]: pylance strict)

    # Weights: Overstow is expensive (ALPHA), Stability is secondary (BETA)
    penalties = PenaltyWeights(overstow=1000.0, stability=10.0)

    metrics, best_ves, cost = run_monte_carlo(args, cargo, vessel, penalties)

    print(f"\n--- FINAL RESULTS ---")
    print(f"Objective Value (Z): {cost:.2f}")
    print(f"Total Rehandles: {metrics['rehandles']}")
    print(
        f"CoG Deviations: Bay={metrics['moments']['bay']:.2f}, Row={metrics['moments']['row']:.2f}, Tier={metrics['moments']['tier']:.2f}")
    print(f"Unstowed Containers: {metrics['left']}")

<<<<<<< HEAD
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
=======
    print("\n--- MCTS RESULT ---")
    print(f"Final Cost: {cost:.0f}")

    # Display simple plan
    count = 0
    if best_ves:
        for s in best_ves.slots.values():
            if s.container:
                count += 1
        print(f"Stowed: {count}/{len(cargo)}")


if __name__ == "__main__":
    main()
>>>>>>> c9faeeb ([Refac]: pylance strict)
