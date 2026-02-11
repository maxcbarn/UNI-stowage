from __future__ import annotations

import argparse
import copy
import math
import random
from typing import List, Optional, Tuple

from common import Vessel, Container, Slot, SlotCoord, calculate_cost, Range

# ==========================================
# 1. HEURISTIC ENGINE (Used for Expansion & Rollout)
# ==========================================


def score_move(vessel: Vessel, container: Container, slot: Slot) -> float:
    score = 0.0
    # Stability: Lower is better
    score -= (slot.tier * container.weight) / 1000.0
    # Balance: Center is better
    dist_row = abs(slot.row - (vessel.rows - 1)/2.0)
    score -= dist_row * (container.weight / 1000.0)
    # Overstowage Prevention
    if slot.tier > 0:
        under = vessel.get_slot_at(
            SlotCoord(slot.bay, slot.row, slot.tier - 1))
        if under:
            below = under.container
            if below:
                if container.dischargePort > below.dischargePort:
                    score -= 10000.0
                if container.weight > below.weight:
                    score -= 5000.0
    return score


# ==========================================
# 3. MONTE CARLO TREE SEARCH (MCTS)
# ==========================================


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
    sorted_cargo = sorted(initial_cargo, key=lambda c: (c.dischargePort, c.weight), reverse=True)

    root = MCTSNode(copy.deepcopy(root_vessel), sorted_cargo)

    best_global_plan = None
    min_global_cost = float('inf')

    # print(f"--- MCTS Start: {iterations} Iterations ---")

    for i in range(iterations):
        node = root

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

        # 2. EXPANSION (Add a new child)
        if not node.is_terminal and not node.is_fully_expanded:
            if node.untried_moves is None:
                node.untried_moves = node.get_legal_moves()

            if node.untried_moves:
                container, slot = node.untried_moves.pop()

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
            #print( f"  > Iter {i}: New Best Cost {cost:.0f} (Left: {len(sim_leftovers)})")

        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    return best_global_plan, min_global_cost

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

    def ri(vals: List[int], f: bool = False):
        if len(vals) == 1:
            return Range(vals[0], vals[0] if f else vals[0]+1)
        return Range(vals[0], vals[1])

    def rf(vals: List[float], f: bool = False):
        if len(vals) == 1:
            return Range(vals[0], vals[0] if f else vals[0]+1)
        return Range(vals[0], vals[1])

    vessel = Vessel(ri(args.bays)(), ri(args.rows)(),
                    ri(args.tiers)())
    cargo = [Container.gen_random(rf(args.weight, True), ri(
        [1, 5])) for _ in range(ri(args.containers)())]

    print(
        f"Initialized MCTS. Ship: {vessel.capacity} slots. Cargo: {len(cargo)} items.")
    best_ves, cost = mcts_search(vessel, cargo, args.iterations)

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
