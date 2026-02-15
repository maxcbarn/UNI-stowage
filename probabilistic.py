from __future__ import annotations

import argparse
import copy
import heapq
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
# 1. SPATIAL INDEX (Optimized)
# ==========================================

class FreeTopsIndex:
    def __init__(self, vessel: Vessel) -> None:
        self.free_tops: Dict[Tuple[int, int], int] = {}
        self._max_tier: Dict[Tuple[int, int], int] = {}

        col_all_tiers: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        col_occupied_max: Dict[Tuple[int, int], int] = defaultdict(lambda: -1)

        for slot in vessel.slots.values():
            col = (slot.bay, slot.row)
            col_all_tiers[col].append(slot.tier)
            if slot.container and slot.tier > col_occupied_max[col]:
                col_occupied_max[col] = slot.tier

        self._heap: List[Tuple[int, int, int]] = []

        for col, tiers in col_all_tiers.items():
            self._max_tier[col] = max(tiers)
            next_free = col_occupied_max[col] + 1
            if next_free <= self._max_tier[col]:
                self.free_tops[col] = next_free
                heapq.heappush(self._heap, (next_free, col[0], col[1]))

    def place(self, bay: int, row: int) -> None:
        col = (bay, row)
        tier = self.free_tops.get(col)
        if tier is None: return
        next_tier = tier + 1
        if next_tier > self._max_tier[col]:
            del self.free_tops[col]
        else:
            self.free_tops[col] = next_tier
            heapq.heappush(self._heap, (next_tier, bay, row))

    def undo(self, bay: int, row: int) -> None:
        col = (bay, row)
        current = self.free_tops.get(col)
        if current is None:
            if col in self._max_tier:
                restored = self._max_tier[col]
                self.free_tops[col] = restored
                heapq.heappush(self._heap, (restored, bay, row))
        else:
            if current > 0:
                self.free_tops[col] = current - 1
                heapq.heappush(self._heap, (current - 1, bay, row))

    def candidates(self) -> List[Tuple[int, int, int]]:
        return [(bay, row, tier) for (bay, row), tier in self.free_tops.items()]

    def best_candidate(self) -> Optional[Tuple[int, int, int]]:
        while self._heap:
            tier, bay, row = heapq.heappop(self._heap)
            col = (bay, row)
            if self.free_tops.get(col) == tier:
                heapq.heappush(self._heap, (tier, bay, row))
                return bay, row, tier
        return None

# ==========================================
# 2. HEURISTIC ENGINE
# ==========================================

def _ucb1(child: MCTSNode, parent_visits: int, exploration_constant: float) -> float:
    if child.visits == 0: return float("inf")
    return (child.value / child.visits) + exploration_constant * math.sqrt(math.log(parent_visits) / child.visits)

def score_move_heavy(vessel: Vessel, container: Container, slot: Slot, crane_cache: Dict[int, int]) -> float:
    score = 0.0
    vcg_proxy = slot.tier * CONTAINER_HEIGHT
    score -= (container.weight * vcg_proxy / 1000.0) * (W_GM_FAIL / 1000.0)

    if slot.tier > 0:
        under = vessel.get_slot_at(SlotCoord(slot.bay, slot.row, slot.tier - 1))
        if under and under.container:
            below = under.container
            if container.dischargePort > below.dischargePort:
                score -= W_REHANDLE
            if container.weight > below.weight:
                score -= W_REHANDLE * 0.5

    center_row = (vessel.rows - 1) / 2.0
    lateral_deviation = abs(slot.row - center_row)
    score -= lateral_deviation * (container.weight / 1000.0) * W_BALANCE
    score -= crane_cache.get(slot.bay, 0) * (W_REHANDLE / 200.0)
    return score

def build_crane_cache(vessel: Vessel) -> Dict[int, int]:
    cache: Dict[int, int] = defaultdict(int)
    for slot in vessel.slots.values():
        if slot.container is not None:
            cache[slot.bay] += 1
    return cache

# ==========================================
# 3. MCTS & BAY ASSIGNMENT
# ==========================================

def assign_preferred_bays(container: Container, free_tops: FreeTopsIndex, bay_discharge_sum: Dict[int, float], crane_cache: Dict[int, int], top_k: int = 3) -> Set[int]:
    free_bays: Set[int] = {bay for (bay, _row) in free_tops.free_tops}
    if not free_bays: return set()
    if top_k >= len(free_bays): return free_bays

    scored: List[Tuple[float, int]] = []
    for bay in free_bays:
        count = crane_cache.get(bay, 0)
        avg_dp = (bay_discharge_sum.get(bay, 0.0) / count) if count > 0 else 0.0
        affinity = -abs(avg_dp - container.dischargePort)
        scored.append((affinity, bay))

    scored.sort(reverse=True)
    return {bay for _, bay in scored[:top_k]}

class MCTSNode:
    def __init__(self, parent: Optional[MCTSNode] = None, move: Optional[Tuple[Container, SlotCoord]] = None, cargo_index: int = 0):
        self.parent = parent
        self.move = move
        self.cargo_index = cargo_index
        self.children: List[MCTSNode] = []
        self.visits: int = 0
        self.value: float = 0.0
        self.untried_moves: Optional[List[Tuple[Container, SlotCoord]]] = None

    @property
    def is_fully_expanded(self) -> bool:
        return self.untried_moves is not None and len(self.untried_moves) == 0

def _place(vessel: Vessel, container: Container, coord: SlotCoord, free_tops: FreeTopsIndex, crane_cache: Dict[int, int], bay_discharge_sum: Dict[int, float]) -> None:
    slot = vessel.get_slot_at(coord)
    vessel.place(container, slot)
    free_tops.place(coord.bay, coord.row)
    crane_cache[coord.bay] = crane_cache.get(coord.bay, 0) + 1
    bay_discharge_sum[coord.bay] = bay_discharge_sum.get(coord.bay, 0.0) + container.dischargePort

def _undo(vessel: Vessel, container: Container, coord: SlotCoord, free_tops: FreeTopsIndex, crane_cache: Dict[int, int], bay_discharge_sum: Dict[int, float]) -> None:
    slot = vessel.get_slot_at(coord)
    slot.container = None
    free_tops.undo(coord.bay, coord.row)
    crane_cache[coord.bay] = max(0, crane_cache.get(coord.bay, 0) - 1)
    bay_discharge_sum[coord.bay] = max(0.0, bay_discharge_sum.get(coord.bay, 0.0) - container.dischargePort)

def mcts_search(root_vessel: Vessel, initial_cargo: List[Container], iterations: int = 100, exploration_constant: float = 0.5) -> Tuple[Optional[Vessel], float]:
    sorted_cargo = sorted(initial_cargo, key=lambda c: (c.dischargePort, c.weight), reverse=True)
    root = MCTSNode(parent=None, move=None, cargo_index=0)
    
    # Track the BEST PARTIAL solution found so far, not just the best complete one.
    best_global_plan: Optional[Vessel] = None
    min_global_cost = float("inf")
    max_stowed_count = -1 

    free_tops = FreeTopsIndex(root_vessel)
    crane_cache = build_crane_cache(root_vessel)
    bay_discharge_sum: Dict[int, float] = defaultdict(float)
    for slot in root_vessel.slots.values():
        if slot.container: bay_discharge_sum[slot.bay] += slot.container.dischargePort

    for _ in range(iterations):
        node = root
        tree_coords: List[SlotCoord] = []
        tree_containers: List[Container] = []

        # 1. SELECTION
        while not node.is_fully_expanded and node.children:
            node = max(node.children, key=lambda c: _ucb1(c, node.visits, exploration_constant))
            if node.move:
                container, coord = node.move
                _place(root_vessel, container, coord, free_tops, crane_cache, bay_discharge_sum)
                tree_coords.append(coord)
                tree_containers.append(container)

        # 2. EXPANSION
        if node.cargo_index < len(sorted_cargo):
            if node.untried_moves is None:
                next_c = sorted_cargo[node.cargo_index]
                preferred_bays = assign_preferred_bays(next_c, free_tops, bay_discharge_sum, crane_cache, top_k=5)
                candidates: List[Tuple[SlotCoord, float]] = []
                for bay, row, tier in free_tops.candidates():
                    if bay not in preferred_bays: continue
                    coord = SlotCoord(bay, row, tier)
                    slot = root_vessel.get_slot_at(coord)
                    if root_vessel.check_hard_constraints(next_c, slot):
                        sc = score_move_heavy(root_vessel, next_c, slot, crane_cache)
                        candidates.append((coord, sc))

                if candidates:
                    tau = 1000.0
                    scores = [x[1] for x in candidates]
                    max_score = max(scores)
                    weights = [math.exp((s - max_score) / tau) for s in scores]
                    k = min(5, len(candidates))
                    selected = random.choices(candidates, weights=weights, k=k * 2)
                    seen: set = set()
                    node.untried_moves = []
                    for coord, _ in selected:
                        if coord not in seen:
                            seen.add(coord)
                            node.untried_moves.append((next_c, coord))
                            if len(node.untried_moves) >= k: break
                else:
                    node.untried_moves = []

            if node.untried_moves:
                container, coord = node.untried_moves.pop()
                _place(root_vessel, container, coord, free_tops, crane_cache, bay_discharge_sum)
                tree_coords.append(coord)
                tree_containers.append(container)
                child_node = MCTSNode(parent=node, move=(container, coord), cargo_index=node.cargo_index + 1)
                node.children.append(child_node)
                node = child_node

        # 3. SIMULATION
        sim_coords: List[SlotCoord] = []
        sim_containers: List[Container] = []
        sim_leftovers: List[Container] = []
        rollout_depth = min(len(sorted_cargo), node.cargo_index + 50)

        for i in range(node.cargo_index, rollout_depth):
            c = sorted_cargo[i]
            result = free_tops.best_candidate()
            if result is not None:
                bay, row, tier = result
                coord = SlotCoord(bay, row, tier)
                slot = root_vessel.get_slot_at(coord)
                if root_vessel.check_hard_constraints(c, slot):
                    _place(root_vessel, c, coord, free_tops, crane_cache, bay_discharge_sum)
                    sim_coords.append(coord)
                    sim_containers.append(c)
                else:
                    sim_leftovers.append(c)
            else:
                sim_leftovers.append(c)

        # 4. BACKPROPAGATION & BEST PLAN UPDATE
        cost = calculate_cost(root_vessel, sim_leftovers)
        stowed_items = node.cargo_index + len(sim_coords)
        
        # FIX: Keep best PARTIAL solution. 
        # If we stowed more items than ever before, save it.
        # If we stowed same amount but cheaper, save it.
        if stowed_items > max_stowed_count or (stowed_items == max_stowed_count and cost < min_global_cost):
            max_stowed_count = stowed_items
            min_global_cost = cost
            best_global_plan = copy.deepcopy(root_vessel)

        # Reward logic
        target_items = rollout_depth
        if len(sim_leftovers) == 0 and stowed_items == target_items:
            reward = 1.0 + 1.0 / (1.0 + cost)
        else:
            ratio = stowed_items / target_items if target_items > 0 else 0
            reward = ratio ** 2

        backprop_node = node
        while backprop_node is not None:
            backprop_node.visits += 1
            backprop_node.value += reward
            backprop_node = backprop_node.parent

        # 5. UNDO
        for coord, container in zip(reversed(sim_coords), reversed(sim_containers)):
            _undo(root_vessel, container, coord, free_tops, crane_cache, bay_discharge_sum)
        for coord, container in zip(reversed(tree_coords), reversed(tree_containers)):
            _undo(root_vessel, container, coord, free_tops, crane_cache, bay_discharge_sum)

    return best_global_plan, min_global_cost

# ==========================================
# 4. ROLLING HORIZON SOLVER
# ==========================================

def solve_rolling_horizon(vessel: Vessel, cargo: List[Container], chunk_size: int = 50, iterations: int = 100, exploration: float = 0.5) -> Tuple[Vessel, float]:
    sorted_cargo = sorted(cargo, key=lambda c: (c.dischargePort, c.weight), reverse=True)
    n_chunks = math.ceil(len(sorted_cargo) / chunk_size)
    print(f"Rolling Horizon: {len(sorted_cargo)} items, {n_chunks} chunks.")

    for i in range(0, len(sorted_cargo), chunk_size):
        chunk = sorted_cargo[i : i + chunk_size]
        
        # 1. Run MCTS
        best_ves, _ = mcts_search(vessel, chunk, iterations, exploration)

        if best_ves is not None:
            vessel = best_ves
            
            # Check if MCTS missed anything in this chunk
            # best_ves might be a partial solution (thanks to the fix above)
            placed_ids = set()
            for s in vessel.slots.values():
                if s.container: placed_ids.add(s.container.id)
            
            leftovers = [c for c in chunk if c.id not in placed_ids]
        else:
            leftovers = chunk # MCTS returned nothing (should be rare with fix)

        # 2. GREEDY FALLBACK for any leftovers from MCTS
        if leftovers:
            print(f"  Warning: Chunk {i//chunk_size+1} has {len(leftovers)} items for Greedy Fallback.")
            ft = FreeTopsIndex(vessel)
            cc = build_crane_cache(vessel)
            
            for c in leftovers:
                best_coord = None
                best_score = -float('inf')
                
                # Scan ALL free tops (exhaustive)
                for bay, row, tier in ft.candidates():
                    coord = SlotCoord(bay, row, tier)
                    slot = vessel.get_slot_at(coord)
                    if vessel.check_hard_constraints(c, slot):
                        s = score_move_heavy(vessel, c, slot, cc)
                        if s > best_score:
                            best_score = s
                            best_coord = coord
                
                if best_coord:
                    slot = vessel.get_slot_at(best_coord)
                    vessel.place(c, slot)
                    ft.place(best_coord.bay, best_coord.row)
                    cc[best_coord.bay] += 1
                else:
                    # If this prints, the ship is PHYSICALLY FULL or constraints are impossible
                    print(f"    CRITICAL: Container {c.id} (P:{c.dischargePort} W:{c.weight}) could not be placed.")

    final_cost = calculate_cost(vessel)
    return vessel, final_cost

def local_search_polish(vessel: Vessel, max_steps: int = 500) -> Tuple[Vessel, float]:
    current_vessel = copy.deepcopy(vessel)
    filled_slots = [s for s in current_vessel.slots.values() if not s.is_free]
    if len(filled_slots) < 2: return current_vessel, calculate_cost(current_vessel)
    current_cost = calculate_cost(current_vessel)
    steps_no_improve = 0

    for _ in range(max_steps):
        if steps_no_improve > 50: break
        s1, s2 = random.sample(filled_slots, 2)
        c1, c2 = s1.container, s2.container
        s1.container, s2.container = None, None
        
        if current_vessel.check_hard_constraints(c1, s2) and current_vessel.check_hard_constraints(c2, s1):
            s1.container, s2.container = c2, c1
            new_cost = calculate_cost(current_vessel)
            if new_cost < current_cost:
                current_cost = new_cost
                steps_no_improve = 0
            else:
                s1.container, s2.container = c1, c2
                steps_no_improve += 1
        else:
            s1.container, s2.container = c1, c2

    return current_vessel, current_cost

def _gen_random_container(cid: int, weight_range: Range, port_range: Range) -> Container:
    return Container(id=cid, weight=round(weight_range(), 1), height=CONTAINER_HEIGHT, dischargePort=int(port_range()))

def main():
    parser = argparse.ArgumentParser(description="MCTS Stowage Solver")
    parser.add_argument("--bays", nargs="+", type=int, default=[5])
    parser.add_argument("--rows", nargs="+", type=int, default=[5])
    parser.add_argument("--tiers", nargs="+", type=int, default=[5])
    parser.add_argument("--containers", nargs="+", type=int, default=[50])
    parser.add_argument("--weight", nargs="+", type=float, default=[1000.0, 30000.0])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exploration", type=float, default=0.5)
    args = parser.parse_args()

    random.seed(args.seed)
    def ri(vals: List[int]) -> Range: return Range(vals[0], vals[0] + 1) if len(vals) == 1 else Range(vals[0], vals[1])
    def rf(vals: List[float]) -> Range: return Range(vals[0], vals[0]) if len(vals) == 1 else Range(vals[0], vals[1])

    vessel = Vessel(ri(args.bays)(), ri(args.rows)(), ri(args.tiers)())
    weight_range = rf(args.weight)
    port_range = ri([1, 5])
    n_cargo = ri(args.containers)()
    cargo = [_gen_random_container(i, weight_range, port_range) for i in range(n_cargo)]

    print(f"Initialized MCTS. Ship: {vessel.capacity} slots. Cargo: {len(cargo)} items.")
    best_ves, cost = solve_rolling_horizon(vessel, cargo, chunk_size=50, iterations=args.iterations, exploration=args.exploration)
    
    if best_ves:
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
            if s.container: count += 1
        print(f"Stowed: {count}/{len(cargo)}")
    else:
        print("No feasible plan found.")

if __name__ == "__main__":
    main()
