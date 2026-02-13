from random import seed
import argparse
import random
from typing import List, Tuple, Optional, Dict

# Import from the unified common library provided in the context
from common import (
    Vessel, Container, Slot, SlotCoord, Range, Numeric,
    calculate_cost, CostReport,
    W_REHANDLE, W_GM_FAIL, W_BALANCE
)


def get_center_out_order(n: int) -> List[int]:
    """
    Returns indices in center-out order (e.g., 5 -> [2, 3, 1, 4, 0]).
    Used for stability (loading from keel up and center out).
    """
    res: List[int] = []
    left, right = (n - 1) // 2, (n - 1) // 2 + 1

    if left >= 0 and left == (n - 1) / 2:
        res.append(int(left))
        left -= 1

    while left >= 0 or right < n:
        if right < n:
            res.append(int(right))
            right += 1
        if left >= 0:
            res.append(int(left))
            left -= 1

    return res if res else list(range(n))


def score_move(vessel: Vessel, container: Container, slot: Slot, bay_density: int) -> float:
    """
    Heuristic Scoring Function.
    Higher Score = Better Move.

    Aligned with Master Cost Function:
    1. Safety (Maximize GM -> Minimize VCG): Heavy items lower.
    2. Efficiency (Minimize Rehandles): W_REHANDLE penalty.
    3. Balance (Crane Workload): Distribute density.
    """
    score = 0.0

    # --- 1. STABILITY (Minimize VCG) ---
    # We use (Tier * Weight) as a proxy for VCG.
    # Penalize placing heavy containers high up.
    score -= (slot.tier * container.weight) * 10.0

    # --- 2. EFFICIENCY (Minimize Rehandles) ---
    if slot.tier > 0:
        slot_below = vessel.get_slot_at(
            SlotCoord(slot.bay, slot.row, slot.tier - 1))

        if slot_below and slot_below.container:
            # Overstow penalty (Critical)
            if container.dischargePort > slot_below.container.dischargePort:
                score -= W_REHANDLE

            # Weight Inversion penalty (Secondary stability check)
            if container.weight > slot_below.container.weight:
                score -= 5000.0

            # Homogeneous Stacking Bonus (Operational efficiency)
            if container.dischargePort == slot_below.container.dischargePort:
                score += 2000.0

    # --- 3. BALANCE (Crane Workload) ---
    # Penalize placing in bays that are already full to encourage spread/balance.
    score -= (bay_density * 500.0)

    return score


def heuristic_solver(containers: List[Container], vessel: Vessel) -> Tuple[Vessel, List[Container]]:
    """
    Constructive Heuristic Solver.
    Places containers one by one into the best available slot based on score_move.
    """
    # [CITATION] Ding & Chou (2015): Sorting by Discharge Port (DESC)
    # This naturally minimizes rehandles and puts heavy items at bottom (if weights differ).
    load_list = sorted(containers, key=lambda c: (
        c.dischargePort, c.weight), reverse=True)

    left_behind: List[Container] = []

    # Equilibrium Logic: Fill from center bay/row outwards
    bay_order = get_center_out_order(vessel.bays)

    # Pre-calculate row orders once
    row_orders = {b: get_center_out_order(
        vessel.rows) for b in range(vessel.bays)}

    # Track bay density for O(1) crane balancing
    bay_counts = {b: 0 for b in range(vessel.bays)}
    for s in vessel.slots.values():
        if s.container:
            bay_counts[s.bay] += 1

    # Main placement loop
    for container in load_list:
        best_slot: Optional[Slot] = None
        best_score = float('-inf')

        # Check every bay in center-out order
        for bay_idx in bay_order:
            # Check every row in center-out order
            for row_idx in row_orders[bay_idx]:
                # Find the lowest empty tier in this column (Stacking constraint)
                target_tier = -1
                for t in range(vessel.tiers):
                    coord = SlotCoord(bay_idx, row_idx, t)
                    slot = vessel.get_slot_at(coord)
                    if slot and slot.is_free:
                        target_tier = t
                        break

                # If column is full, skip
                if target_tier == -1:
                    continue

                # Evaluate this specific move
                candidate_slot = vessel.get_slot_at(
                    SlotCoord(bay_idx, row_idx, target_tier))

                # Double check hard constraints
                if candidate_slot and vessel.check_hard_constraints(container, candidate_slot):
                    # Use local scoring function (NOT global calculate_cost)
                    score = score_move(vessel, container,
                                       candidate_slot, bay_counts[bay_idx])

                    if score > best_score:
                        best_score = score
                        best_slot = candidate_slot

        # Commit the best move found
        if best_slot:
            vessel.place(container, best_slot)
            bay_counts[best_slot.bay] += 1
        else:
            left_behind.append(container)

    return vessel, left_behind


def print_fitness_report(vessel: Vessel) -> None:
    """
    Uses the unified CostReport from common.py to display results.
    """
    report = CostReport(vessel)

    print("\n--- UNIFIED PHYSICS & FITNESS REPORT ---")
    print(f"Safety (GM):            {report.gm:.4f} m  (Target: >= 1.0m)")
    print(f"Efficiency (Rehandles): {report.rehandles}")
    print(f"Trim (Bay Moment):      {report.bay_moment:.4f}")
    print(f"List (Row Moment):      {report.row_moment:.4f}")
    print(f"Vertical Moment (VCG):  {report.tier_moment:.4f}")

    print(f"--> MASTER COST SCORE:  {report.total_cost:.4f}")


def gen_case(
        bays: Range[int],
        rows: Range[int],
        tiers: Range[int],
        container_amount: Range[int],
        weight: Range[float],
        ports: Range[int],
) -> None:

    # Use keyword args to avoid dataclass ordering issues
    vessel = Vessel(bays=bays(), rows=rows(), tiers=tiers())
    cargo_amt = container_amount()

    print(
        f"\n[GENERATION] Ship: {vessel.bays}x{vessel.rows}x{vessel.tiers} (Cap: {vessel.capacity}) | Cargo: {cargo_amt}")

    cargo = [Container.gen_random(weight, ports) for _ in range(cargo_amt)]

    # Run Solver
    vessel, left_behind = heuristic_solver(cargo, vessel)

    print("\n--- STOWAGE RESULTS ---")
    stowed_count = vessel.containerAmount
    # Handle division by zero if empty list
    pct = (stowed_count / len(cargo)) * 100 if cargo else 0.0
    print(f"Stowed: {stowed_count}/{len(cargo)} ({pct:.1f}%)")

    if left_behind:
        print(f"[WARNING] {len(left_behind)} containers left on dock!")
        # Calculate cost including leftovers penalty
        total_cost = calculate_cost(vessel, left_behind)
        print(f"--> COST WITH LEFTOVERS: {total_cost:.4f}")
    else:
        print("[SUCCESS] 100% Stowage Achieved.")

    print_fitness_report(vessel)


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--bays", nargs="+", type=int, default=[5])
    parser.add_argument("--rows", nargs="+", type=int, default=[5])
    parser.add_argument("--tiers", nargs="+", type=int, default=[5])
    parser.add_argument("--containers", nargs="+", type=int, default=[50, 100])
    parser.add_argument("--weight", nargs="+", type=float,
                        default=[1000.0, 30000.0])
    parser.add_argument("--ports", nargs="+", type=int, default=[1, 5])
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    def to_range(values: List[Numeric], is_float: bool = False) -> Range[Numeric]:
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
        random.seed(args.seed)
        print(f"--- SEED: {args.seed} ---")

    try:
        gen_case(
            bays=to_range(args.bays),
            rows=to_range(args.rows),
            tiers=to_range(args.tiers),
            container_amount=to_range(args.containers),
            weight=to_range(args.weight, is_float=True),
            ports=to_range(args.ports),
        )
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
