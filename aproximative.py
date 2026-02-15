from random import seed
import argparse
import random
import copy
from typing import List, Tuple, Optional, Dict

# Import from the unified common library
from common import (
    Vessel, Container, Slot, SlotCoord, Range, Numeric,
    calculate_cost, calculate_gm,
    W_REHANDLE, W_STACK_BONUS, W_CRANE_BALANCE,
    W_WEIGHT_INVERSION, W_MOMENT_TIER, W_MOMENT_ROW
)

HEAVY_THRESHOLD = 20.0

# --- HELPER: Fast Candidate Finding ---
def get_candidate_slots(vessel: Vessel, bay_idx: int) -> List[Slot]:
    """
    Performance Fix: Only return the *topmost* valid slot for each row in the bay.
    Complexity: O(Rows) instead of O(Rows * Tiers).
    """
    candidates = []
    for r in range(vessel.rows):
        for t in range(vessel.tiers):
            coord = SlotCoord(bay_idx, r, t)
            slot = vessel.slots.get(coord)
            # We look for the first empty slot. 
            # Gravity constraint is implicitly satisfied because we scan from t=0 up.
            if slot and slot.container is None:
                candidates.append(slot)
                break 
    return candidates


def get_center_out_order(n: int) -> List[int]:
    """Indices in center-out order (e.g., 5 -> [2, 3, 1, 4, 0])."""
    res: List[int] = []
    left, right = (n - 1) // 2, (n - 1) // 2 + 1
    if left >= 0 and left == (n - 1) / 2:
        res.append(int(left)); left -= 1
    while left >= 0 or right < n:
        if right < n: res.append(int(right)); right += 1
        if left >= 0: res.append(int(left)); left -= 1
    return res if res else list(range(n))


def score_move(vessel: Vessel, container: Container, slot: Slot, bay_density: int) -> float:
    """Weighted scoring function using unified constants."""
    score = 0.0
    
    # 1. STABILITY (Normalized by 1000kg to keep score manageable)
    # Penalize high VCG
    score -= (slot.tier * container.weight / 1000.0) * W_MOMENT_TIER
    
    # Penalize List (Off-center)
    center_row = (vessel.rows - 1) / 2.0
    dist_row = abs(slot.row - center_row)
    score -= (dist_row * container.weight / 1000.0) * W_MOMENT_ROW

    # 2. OPERATIONAL LOGIC
    # Check the slot immediately below
    if slot.tier > 0:
        coord_below = SlotCoord(slot.bay, slot.row, slot.tier - 1)
        slot_below = vessel.slots.get(coord_below)
        
        if slot_below and slot_below.container:
            below = slot_below.container
            
            # [CRITICAL] Rehandle Check
            if container.dischargePort > below.dischargePort:
                score -= W_REHANDLE 
            
            # [CRITICAL] Weight Inversion Check
            if container.weight > below.weight:
                score -= W_WEIGHT_INVERSION

            # [BONUS] Homogeneous Stack
            if container.dischargePort == below.dischargePort:
                score += W_STACK_BONUS

    # 3. CRANE BALANCING
    score -= (bay_density * W_CRANE_BALANCE)
    
    return score


def solve_with_strategy(
    containers: List[Container], 
    vessel_template: Vessel, 
    strategy: str
) -> Tuple[Vessel, List[Container], List[Tuple[Container, Slot]], float]:
    
    # Create fresh copy for this run
    vessel = copy.deepcopy(vessel_template)
    
    # --- SMART SORTING STRATEGIES ---
    if strategy == "STABILITY":
        # Strict Heavy -> Light. 
        # Within Heavy: Sort by Port DESC (minimize rehandles among heavies)
        heavy = sorted([c for c in containers if c.weight >= HEAVY_THRESHOLD], 
                       key=lambda c: (c.dischargePort, c.weight), reverse=True)
        light = sorted([c for c in containers if c.weight < HEAVY_THRESHOLD], 
                       key=lambda c: (c.dischargePort, c.weight), reverse=True)
        load_list = heavy + light

    elif strategy == "DENSITY":
        # Pure Logic: Discharge Port is King.
        load_list = sorted(containers, key=lambda c: (c.dischargePort, c.weight), reverse=True)
        
    elif strategy == "HYBRID":
        # [FIX] Multiplier increased to 1000 to prevent Weight from overriding Port.
        # Max weight ~30t. 30 < 1000. Logic holds.
        load_list = sorted(containers, key=lambda c: (c.dischargePort * 1000 + c.weight), reverse=True)

    else:
        load_list = list(containers)

    plan: List[Tuple[Container, Slot]] = []
    left_behind: List[Container] = []
    
    bay_order = get_center_out_order(vessel.bays)
    bay_counts = {b: 0 for b in range(vessel.bays)}

    # --- FAST PLACEMENT LOOP ---
    for container in load_list:
        best_slot = None
        best_score = float('-inf')

        for bay_idx in bay_order:
            # [OPTIMIZATION] Only check top-most valid slots
            candidates = get_candidate_slots(vessel, bay_idx)
            
            for slot in candidates:
                # Score using the unified W_ constants
                s = score_move(vessel, container, slot, bay_counts[bay_idx])
                
                if s > best_score:
                    best_score = s
                    best_slot = slot

        if best_slot:
            vessel.place(container, best_slot)
            plan.append((container, best_slot))
            bay_counts[best_slot.bay] += 1
        else:
            left_behind.append(container)

    cost = calculate_cost(vessel, left_behind)
    return vessel, left_behind, plan, cost


def heuristic_solver(containers: List[Container], vessel: Vessel) -> Tuple[Vessel, List[Container]]:
    """
    Adaptive Solver: Tries strategies and picks the best.
    """
    # Order matters: DENSITY is usually the cleanest for rehandles.
    strategies = ["DENSITY", "STABILITY", "HYBRID"]
    
    best_res = None
    best_leftovers = []
    best_cost = float('inf')

    for strat in strategies:
        v_res, l_res, plan, cost = solve_with_strategy(containers, vessel, strat)
        
        # print(f"Strategy {strat}: Cost={cost:.0f}, Leftovers={len(l_res)}")
        
        if cost < best_cost:
            best_cost = cost
            best_res = v_res
            best_leftovers = l_res
            
        # Early exit if solution is physically perfect
        # (Zero leftovers, Zero rehandles, Safe GM)
        if len(l_res) == 0 and v_res.calculate_rehandles() == 0:
             # Calculate GM only if operations are perfect to save time
             if calculate_gm(v_res) >= 1.0:
                 break

    return best_res, best_leftovers
