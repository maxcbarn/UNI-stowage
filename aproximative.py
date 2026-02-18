from random import seed
import argparse
import random
import copy
from typing import List, Tuple, Optional, Dict


from common import (
    Vessel, Container, Slot, SlotCoord, Range, Numeric,
    calculate_cost, calculate_gm,
    W_REHANDLE, W_STACK_BONUS, W_CRANE_BALANCE,
    W_WEIGHT_INVERSION, W_MOMENT_TIER, W_MOMENT_ROW
)

HEAVY_THRESHOLD = 20.0


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
    
    
    
    score -= (slot.tier * container.weight / 1000.0) * W_MOMENT_TIER
    
    
    center_row = (vessel.rows - 1) / 2.0
    dist_row = abs(slot.row - center_row)
    score -= (dist_row * container.weight / 1000.0) * W_MOMENT_ROW

    
    
    if slot.tier > 0:
        coord_below = SlotCoord(slot.bay, slot.row, slot.tier - 1)
        slot_below = vessel.slots.get(coord_below)
        
        if slot_below and slot_below.container:
            below = slot_below.container
            
            
            if container.dischargePort > below.dischargePort:
                score -= W_REHANDLE 
            
            
            if container.weight > below.weight:
                score -= W_WEIGHT_INVERSION

            
            if container.dischargePort == below.dischargePort:
                score += W_STACK_BONUS

    
    score -= (bay_density * W_CRANE_BALANCE)
    
    return score


def solve_with_strategy(
    containers: List[Container], 
    vessel_template: Vessel, 
    strategy: str
) -> Tuple[Vessel, List[Container], List[Tuple[Container, Slot]], float]:
    
    
    vessel = copy.deepcopy(vessel_template)
    
    
    if strategy == "STABILITY":
        
        
        heavy = sorted([c for c in containers if c.weight >= HEAVY_THRESHOLD], 
                       key=lambda c: (c.dischargePort, c.weight), reverse=True)
        light = sorted([c for c in containers if c.weight < HEAVY_THRESHOLD], 
                       key=lambda c: (c.dischargePort, c.weight), reverse=True)
        load_list = heavy + light

    elif strategy == "DENSITY":
        
        load_list = sorted(containers, key=lambda c: (c.dischargePort, c.weight), reverse=True)
        
    elif strategy == "HYBRID":
        
        
        load_list = sorted(containers, key=lambda c: (c.dischargePort * 1000 + c.weight), reverse=True)

    else:
        load_list = list(containers)

    plan: List[Tuple[Container, Slot]] = []
    left_behind: List[Container] = []
    
    bay_order = get_center_out_order(vessel.bays)
    bay_counts = {b: 0 for b in range(vessel.bays)}

    
    for container in load_list:
        best_slot = None
        best_score = float('-inf')

        for bay_idx in bay_order:
            
            candidates = get_candidate_slots(vessel, bay_idx)
            
            for slot in candidates:
                
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
    
    strategies = ["DENSITY", "STABILITY", "HYBRID"]
    
    best_res = None
    best_leftovers = []
    best_cost = float('inf')

    for strat in strategies:
        v_res, l_res, plan, cost = solve_with_strategy(containers, vessel, strat)
        
        
        
        if cost < best_cost:
            best_cost = cost
            best_res = v_res
            best_leftovers = l_res
            
        
        
        if len(l_res) == 0 and v_res.calculate_rehandles() == 0:
             
             if calculate_gm(v_res) >= 1.0:
                 break

    return best_res, best_leftovers
