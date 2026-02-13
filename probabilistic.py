from __future__ import annotations

import argparse
import copy
import math
import random
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from common import (
    Vessel, Container, Slot, SlotCoord, Range,
    calculate_cost, calculate_gm,
    CONTAINER_HEIGHT, MIN_GM,
    W_REHANDLE, W_GM_FAIL, W_BALANCE,
)

# ==========================================
# 1. SPATIAL INDEX
# ==========================================


class FreeTopsIndex:
    """
    OPT-1: Replaces the O(B×R×T) slot scan with an O(B×R) structure.

    Invariant: free_tops[(bay, row)] == the lowest free tier in that column,
    or the key is absent if the column is full.

    Because stacking is always bottom-up, the only valid placement per column
    is always exactly this one tier — no need to check any other tier in the
    column, and no need to visit occupied slots at all.
    """

    def __init__(self, vessel: Vessel) -> None:
        self.free_tops: Dict[Tuple[int, int], int] = {}
        self._max_tier: Dict[Tuple[int, int], int] = {}

        col_tiers: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for slot in vessel.slots.values():
            col = (slot.bay, slot.row)
            col_tiers[col].append(slot.tier)

        for col, tiers in col_tiers.items():
            self._max_tier[col] = max(tiers)
            self.free_tops[col] = 0  # all empty at start

    def place(self, bay: int, row: int) -> None:
        """Mark the current free top as occupied and advance the pointer."""
        col = (bay, row)
        tier = self.free_tops.get(col)
        if tier is None:
            return
        next_tier = tier + 1
        if next_tier > self._max_tier[col]:
            del self.free_tops[col]  # column now full
        else:
            self.free_tops[col] = next_tier

    def undo(self, bay: int, row: int) -> None:
        """Retreat the free-top pointer (undo a placement)."""
        col = (bay, row)
        current = self.free_tops.get(col)
        if current is None:
            self.free_tops[col] = self._max_tier[col]
        else:
            self.free_tops[col] = current - 1

    def candidates(self) -> List[Tuple[int, int, int]]:
        """Return (bay, row, tier) for every column that has a free slot."""
        return [(bay, row, tier) for (bay, row), tier in self.free_tops.items()]


# ==========================================
# 2. HEURISTIC ENGINE (TIERED)
# ==========================================


def score_move_heavy(
    vessel: Vessel,
    container: Container,
    slot: Slot,
    crane_cache: Dict[int, int],  # OPT-2: pre-computed per-bay counts
) -> float:
    """
    HEAVY Heuristic: Used for Tree Expansion.

    Scores a candidate placement using the same cost components as
    calculate_cost() from common.py, scaled by the same unified weights:

      - GM / Stability (W_GM_FAIL):  penalises placing heavy containers high,
        which raises VCG and reduces GM.
      - Rehandles (W_REHANDLE):      penalises discharge-port inversions and
        weight inversions directly below the new slot.
      - Balance (W_BALANCE):         penalises deviation from the lateral
        centre (row moment contribution of this container).
      - Crane Intensity (OPT-2):     penalises already-busy bays to spread
        load, scaled proportionally to W_REHANDLE.
    """
    score = 0.0

    # ------------------------------------------------------------------
    # 1. Stability — GM contribution (W_GM_FAIL)
    #    Placing a container at tier t raises VCG by approximately:
    #      ΔVCG ≈ weight * (lightship_vcg + tier * CONTAINER_HEIGHT) / disp
    #    We approximate the marginal cost as proportional to tier * weight.
    #    Higher tiers → higher VCG → lower GM → higher penalty.
    # ------------------------------------------------------------------
    vcg_proxy = slot.tier * CONTAINER_HEIGHT  # metres above baseline
    score -= (container.weight * vcg_proxy / 1000.0) * (W_GM_FAIL / 1000.0)

    # ------------------------------------------------------------------
    # 2. Rehandle penalty — discharge-port and weight inversions (W_REHANDLE)
    #    Mirrors vessel.calculate_rehandles() for the single slot below.
    # ------------------------------------------------------------------
    if slot.tier > 0:
        under = vessel.get_slot_at(
            SlotCoord(slot.bay, slot.row, slot.tier - 1))
        if under and under.container:
            below = under.container
            # Discharge-port inversion: this container discharges later → overstow
            if container.dischargePort > below.dischargePort:
                score -= W_REHANDLE
            # Weight inversion: heavier on top of lighter → unsafe / costly
            if container.weight > below.weight:
                score -= W_REHANDLE * 0.5

    # ------------------------------------------------------------------
    # 3. Balance — lateral (row) moment contribution (W_BALANCE)
    #    Penalises placing weight far from the transverse centre.
    # ------------------------------------------------------------------
    center_row = (vessel.rows - 1) / 2.0
    lateral_deviation = abs(slot.row - center_row)
    score -= lateral_deviation * (container.weight / 1000.0) * W_BALANCE

    # ------------------------------------------------------------------
    # 4. Crane Intensity — OPT-2: O(1) lookup (proportional to W_REHANDLE)
    #    Spreading containers across bays reduces crane congestion.
    # ------------------------------------------------------------------
    score -= crane_cache.get(slot.bay, 0) * (W_REHANDLE / 200.0)

    return score


def build_crane_cache(vessel: Vessel) -> Dict[int, int]:
    """OPT-2: Build bay->count dict once rather than scanning on every call."""
    cache: Dict[int, int] = defaultdict(int)
    for slot in vessel.slots.values():
        if slot.container is not None:
            cache[slot.bay] += 1
    return cache


# ==========================================
# 3. BAY PRE-ASSIGNMENT (OPT-4)
# ==========================================


def assign_preferred_bays(
    container: Container,
    vessel: Vessel,
    free_tops: FreeTopsIndex,
    top_k: int = 3,
) -> Set[int]:
    """
    OPT-4: Restrict MCTS candidates to the top-k bays most aligned with
    the container's discharge port, dramatically shrinking the branching
    factor before scoring begins.

    Each bay is scored by how close its current average discharge port is
    to the container's discharge port. Empty bays are treated as neutral.
    """
    bay_scores: Dict[int, float] = defaultdict(float)
    bay_counts: Dict[int, int] = defaultdict(int)

    for slot in vessel.slots.values():
        if slot.container is not None:
            bay_scores[slot.bay] += slot.container.dischargePort
            bay_counts[slot.bay] += 1

    free_bays: Set[int] = {bay for (bay, _row) in free_tops.free_tops}

    scored: List[Tuple[float, int]] = []
    for bay in free_bays:
        if bay_counts[bay] > 0:
            avg_dp = bay_scores[bay] / bay_counts[bay]
            affinity = -abs(avg_dp - container.dischargePort)
        else:
            affinity = 0.0  # empty bay: neutral
        scored.append((affinity, bay))

    scored.sort(reverse=True)
    return {bay for _, bay in scored[:top_k]}


# ==========================================
# 4. MONTE CARLO TREE SEARCH
# ==========================================


class MCTSNode:
    def __init__(
        self,
        parent: Optional[MCTSNode] = None,
        move: Optional[Tuple[Container, SlotCoord]] = None,
        cargo_index: int = 0,
    ):
        self.parent = parent
        self.move = move  # (Container, SlotCoord)
        self.cargo_index = cargo_index
        self.children: List[MCTSNode] = []
        self.visits: int = 0
        self.value: float = 0.0
        self.untried_moves: Optional[List[Tuple[Container, SlotCoord]]] = None

    @property
    def is_fully_expanded(self) -> bool:
        return self.untried_moves is not None and len(self.untried_moves) == 0


def _place(
    vessel: Vessel,
    container: Container,
    coord: SlotCoord,
    free_tops: FreeTopsIndex,
    crane_cache: Dict[int, int],
) -> None:
    """Place a container and update both indexes atomically."""
    slot = vessel.get_slot_at(coord)
    vessel.place(container, slot)
    free_tops.place(coord.bay, coord.row)
    crane_cache[coord.bay] = crane_cache.get(coord.bay, 0) + 1


def _undo(
    vessel: Vessel,
    coord: SlotCoord,
    free_tops: FreeTopsIndex,
    crane_cache: Dict[int, int],
) -> None:
    """Remove a container and roll back both indexes atomically."""
    slot = vessel.get_slot_at(coord)
    slot.container = None
    free_tops.undo(coord.bay, coord.row)
    crane_cache[coord.bay] = max(0, crane_cache.get(coord.bay, 0) - 1)


def local_search_polish(vessel: Vessel, max_steps: int = 500) -> Tuple[Vessel, float]:
    """
    Post-processing Hill Climbing: swap pairs to squeeze extra efficiency.

    Uses calculate_cost() from common.py directly, which accounts for
    GM safety, rehandles, and balance moments.
    """
    current_vessel = copy.deepcopy(vessel)
    filled_slots = [s for s in current_vessel.slots.values() if not s.is_free]

    if len(filled_slots) < 2:
        return current_vessel, calculate_cost(current_vessel, [])

    current_cost = calculate_cost(current_vessel, [])
    steps_without_improv = 0

    for _ in range(max_steps):
        if steps_without_improv > 50:
            break

        s1 = random.choice(filled_slots)
        s2 = random.choice(filled_slots)
        if s1 == s2:
            continue

        c1, c2 = s1.container, s2.container
        s1.container = None
        s2.container = None

        if current_vessel.check_hard_constraints(
            c1, s2
        ) and current_vessel.check_hard_constraints(c2, s1):
            s1.container = c2
            s2.container = c1
            new_cost = calculate_cost(current_vessel, [])
            if new_cost < current_cost:
                current_cost = new_cost
                steps_without_improv = 0
            else:
                s1.container = c1
                s2.container = c2
                steps_without_improv += 1
        else:
            s1.container = c1
            s2.container = c2

    return current_vessel, current_cost


def mcts_search(
    root_vessel: Vessel,
    initial_cargo: List[Container],
    iterations: int = 1000,
    exploration_constant: float = 0.5,
) -> Tuple[Optional[Vessel], float]:
    """
    Monte Carlo Tree Search stowage solver.

    Evaluation uses calculate_cost() from common.py, which combines:
      - GM safety penalty     (W_GM_FAIL)
      - Rehandle count        (W_REHANDLE)
      - Bay/row balance       (W_BALANCE)
      - Unloaded cargo        (W_LEFTOVER)
    """
    sorted_cargo = sorted(
        initial_cargo, key=lambda c: (c.dischargePort, c.weight), reverse=True
    )

    root = MCTSNode(parent=None, move=None, cargo_index=0)
    best_global_plan: Optional[Vessel] = None
    min_global_cost = float("inf")

    # OPT-1 & OPT-2: Build shared indexes once; update incrementally.
    free_tops = FreeTopsIndex(root_vessel)
    crane_cache = build_crane_cache(root_vessel)

    for _ in range(iterations):
        node = root
        tree_coords: List[SlotCoord] = []

        # -------------------------------------------------------------- #
        # 1. SELECTION                                                     #
        # -------------------------------------------------------------- #
        while not node.is_fully_expanded and node.children:
            def ucb1(child: MCTSNode) -> float:
                if child.visits == 0:
                    return float("inf")
                return (child.value / child.visits) + exploration_constant * math.sqrt(
                    math.log(node.visits) / child.visits
                )

            node = max(node.children, key=ucb1)
            if node.move:
                container, coord = node.move
                _place(root_vessel, container, coord, free_tops, crane_cache)
                tree_coords.append(coord)

        # -------------------------------------------------------------- #
        # 2. EXPANSION                                                     #
        # -------------------------------------------------------------- #
        if node.cargo_index < len(sorted_cargo):
            if node.untried_moves is None:
                next_c = sorted_cargo[node.cargo_index]

                # OPT-4: Restrict to top-k preferred bays first
                preferred_bays = assign_preferred_bays(
                    next_c, root_vessel, free_tops, top_k=3
                )

                # OPT-1: Only visit one slot per column (the free top)
                candidates: List[Tuple[SlotCoord, float]] = []
                for bay, row, tier in free_tops.candidates():
                    if bay not in preferred_bays:
                        continue
                    coord = SlotCoord(bay, row, tier)
                    slot = root_vessel.get_slot_at(coord)
                    if root_vessel.check_hard_constraints(next_c, slot):
                        # OPT-2: Pass crane_cache — no full vessel scan
                        sc = score_move_heavy(
                            root_vessel, next_c, slot, crane_cache
                        )
                        candidates.append((coord, sc))

                if candidates:
                    tau = 1000.0
                    scores = [x[1] for x in candidates]
                    max_score = max(scores)
                    weights = [math.exp((s - max_score) / tau) for s in scores]
                    k = min(5, len(candidates))
                    selected = random.choices(
                        candidates, weights=weights, k=k * 2)

                    seen: set = set()
                    node.untried_moves = []
                    for coord, _ in selected:
                        if coord not in seen:
                            seen.add(coord)
                            node.untried_moves.append((next_c, coord))
                            if len(node.untried_moves) >= k:
                                break
                else:
                    node.untried_moves = []

            if node.untried_moves:
                container, coord = node.untried_moves.pop()
                _place(root_vessel, container, coord, free_tops, crane_cache)
                tree_coords.append(coord)

                child_node = MCTSNode(
                    parent=node,
                    move=(container, coord),
                    cargo_index=node.cargo_index + 1,
                )
                node.children.append(child_node)
                node = child_node

        # -------------------------------------------------------------- #
        # 3. SIMULATION — OPT-1/3: O(B×R) scan, no scoring loop needed   #
        # -------------------------------------------------------------- #
        sim_coords: List[SlotCoord] = []
        sim_leftovers: List[Container] = []

        for i in range(node.cargo_index, len(sorted_cargo)):
            c = sorted_cargo[i]

            # OPT-3: score_move_light == -tier, so the best slot is simply
            # the free-top column with the minimum tier value. No scoring
            # loop required — one pass to find the min-tier free column.
            best_coord: Optional[SlotCoord] = None
            best_tier = float("inf")

            for bay, row, tier in free_tops.candidates():
                if tier < best_tier:
                    coord = SlotCoord(bay, row, tier)
                    slot = root_vessel.get_slot_at(coord)
                    if root_vessel.check_hard_constraints(c, slot):
                        best_tier = tier
                        best_coord = coord

            if best_coord:
                _place(root_vessel, c, best_coord, free_tops, crane_cache)
                sim_coords.append(best_coord)
            else:
                sim_leftovers.append(c)

        # -------------------------------------------------------------- #
        # 4. EVALUATE & RECORD BEST                                        #
        # Uses calculate_cost() from common.py (GM + rehandles + balance  #
        # + leftovers), ensuring MCTS optimises the same objective as the  #
        # final reported score.                                            #
        # -------------------------------------------------------------- #
        cost = calculate_cost(root_vessel, sim_leftovers)

        total_items = len(sorted_cargo)
        stowed_items = node.cargo_index + len(sim_coords)
        is_feasible = (len(sim_leftovers) == 0) and (
            stowed_items == total_items)

        if is_feasible and cost < min_global_cost:
            min_global_cost = cost
            best_global_plan = copy.deepcopy(root_vessel)

        # -------------------------------------------------------------- #
        # 5. BACKPROPAGATION                                               #
        # -------------------------------------------------------------- #
        if is_feasible:
            reward = 1.0 + 1.0 / (1.0 + cost)
        else:
            ratio = stowed_items / total_items
            reward = ratio ** 2

        backprop_node = node
        while backprop_node is not None:
            backprop_node.visits += 1
            backprop_node.value += reward
            backprop_node = backprop_node.parent

        # -------------------------------------------------------------- #
        # 6. UNDO ALL MOVES                                                #
        # -------------------------------------------------------------- #
        for coord in reversed(sim_coords):
            _undo(root_vessel, coord, free_tops, crane_cache)

        for coord in reversed(tree_coords):
            _undo(root_vessel, coord, free_tops, crane_cache)

    return best_global_plan, min_global_cost


# ==========================================
# 5. CLI RUNNER
# ==========================================


def _gen_random_container(cid: int, weight_range: Range, port_range: Range) -> Container:
    """Generate a random Container compatible with common.py's Container dataclass."""
    return Container(
        id=cid,
        weight=round(weight_range(), 1),
        dischargePort=int(port_range()),
    )


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
    parser.add_argument(
        "--exploration",
        type=float,
        default=0.5,
        help="UCB1 exploration constant (lower = more exploitation)",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    def ri(vals: List[int]) -> Range:
        return Range(vals[0], vals[0] + 1) if len(vals) == 1 else Range(vals[0], vals[1])

    def rf(vals: List[float]) -> Range:
        return Range(vals[0], vals[0]) if len(vals) == 1 else Range(vals[0], vals[1])

    vessel = Vessel(ri(args.bays)(), ri(args.rows)(), ri(args.tiers)())
    weight_range = rf(args.weight)
    port_range = ri([1, 5])
    n_cargo = ri(args.containers)()
    cargo = [
        _gen_random_container(i, weight_range, port_range)
        for i in range(n_cargo)
    ]

    print(
        f"Initialized MCTS. Ship: {vessel.capacity} slots. Cargo: {len(cargo)} items.")

    best_ves, cost = mcts_search(
        vessel, cargo, args.iterations, args.exploration)

    if best_ves:
        # Run local search polish pass to tighten the solution
        best_ves, cost = local_search_polish(best_ves)

    print("\n--- MCTS RESULT ---")
    print(f"Final Cost: {cost:.0f}")

    gm = calculate_gm(best_ves) if best_ves else float("nan")
    rehandles = best_ves.calculate_rehandles() if best_ves else 0
    print(f"GM: {gm:.2f} m  (min {MIN_GM} m)")
    print(f"Rehandles: {rehandles}")

    count = 0
    if best_ves:
        for s in best_ves.slots.values():
            if s.container:
                count += 1
        print(f"Stowed: {count}/{len(cargo)}")
    else:
        print("No feasible plan found.")


if __name__ == "__main__":
    main()
