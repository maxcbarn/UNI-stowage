from random import seed
from typing import List, Tuple, TypedDict, Optional, Dict

from common import Vessel, Container, Slot, SlotCoord, Range, Numeric
from itertools import product
import argparse

# --- AMBROSINO & SCIOMACHEN WEIGHTED SCORING CONFIGURATION ---
# References:
# [1] Ambrosino et al. (2004): "Stowing a containership: the Master Bay Plan problem"
# [2] Sciomachen & Tanfani (2003): "Problems and methods for stowage planning"

# Weight for avoiding a rehandle (Overstow).
W_REHANDLE = 10_000.0

# Weight for placing heavy containers on top of light ones.
W_WEIGHT_INVERSION = 5_000.0

# Bonus for "Homogeneous Stacking" (Matching Discharge Ports).
W_STACK_BONUS = 5_000.0

# Sciomachen (2003) notes that distributing containers across bays allows
# parallel crane operations, reducing total port stay time.
W_CRANE_BALANCE = 500.0

# Weights for Center of Gravity (Stability) Penalties.
W_MOMENT_TIER = 1.0  # Vertical Moment (Minimizing VCG)
W_MOMENT_ROW = 1.0   # Transverse Moment (Minimizing List/TCG)
# -----------------------------------------------------------


class PhysicsUtils:
    class Moments(TypedDict):
        bay: float
        row: float
        tier: float

    @staticmethod
    def calculate_rehandles(vessel: Vessel) -> int:
        """Counts how many times a later-port container sits on an earlier-port container."""
        total = 0
        for b, r in product(range(vessel.bays), range(vessel.rows)):
            stack: List[Optional[Slot]] = [vessel.get_slot_at(SlotCoord(b, r, t))
                                           for t in range(vessel.tiers)]
            filled: List[Slot] = [
                s for s in stack if s is not None and s.container is not None]

            for i in range(1, len(filled)):
                below = filled[i-1].container
                above = filled[i].container
                if above and below and above.dischargePort > below.dischargePort:
                    total += 1
        return total

    @staticmethod
    def calculate_moments(vessel: Vessel) -> Moments:
        """Calculates Center of Gravity deviations."""
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


def get_center_out_order(n: int) -> List[int]:
    """
    [CITATION] Improved search order inspired by Ambrosino et al. (2004).
    Addresses 'Equilibrium Constraints' via center-out filling.
    """
    res: List[int] = []
    left, right = (n - 1) // 2, (n - 1) // 2 + 1

    if left >= 0 and left == (n - 1) / 2:
        res.append(left)
        left -= 1

    while left >= 0 or right < n:
        if right < n:
            res.append(right)
            right += 1
        if left >= 0:
            res.append(left)
            left -= 1

    if not res and n > 0:
        return list(range(n))
    return res


def score_move(vessel: Vessel, container: Container, slot: Slot, bay_density: int) -> float:
    """
    Assigns a score to a placement using the Ambrosino Weighted System.
    Higher Score = Better Move.
    [UPDATED] Now accepts 'bay_density' to optimize crane split without O(N) loops.
    """
    score = 0.0
    slot_below: Optional[Slot] = None

    if slot.tier > 0:
        slot_below = vessel.get_slot_at(
            SlotCoord(slot.bay, slot.row, slot.tier - 1))

    # --- 1. STABILITY COMPONENT (Ambrosino et al.) ---
    score -= (slot.tier * container.weight / 1000.0) * W_MOMENT_TIER

    center_row = (vessel.rows - 1) / 2.0
    dist_row = abs(slot.row - center_row)
    score -= (dist_row * container.weight / 1000.0) * W_MOMENT_ROW

    # --- 2. OPERATIONAL COMPONENT (Ambrosino et al.) ---
    if slot_below and slot_below.container:
        # [CITATION] Minimize Rehandles
        if container.dischargePort > slot_below.container.dischargePort:
            score -= W_REHANDLE

        # [CITATION] Minimize Weight Inversion
        if container.weight > slot_below.container.weight:
            score -= W_WEIGHT_INVERSION

        # [CITATION] Homogeneous Stacks (Sciomachen & Tanfani)
        if container.dischargePort == slot_below.container.dischargePort:
            score += W_STACK_BONUS

    # --- 3. CRANE WORKLOAD BALANCING (New Integration) ---
    # [CITATION] Distribute cargo to allow parallel crane moves (Sciomachen 2003)
    # Penalize bays that are already dense relative to others.
    score -= (bay_density * W_CRANE_BALANCE)

    return score


def heuristic_solver(containers: List[Container], vessel: Vessel) -> Tuple[Vessel, List[Container]]:
    """
    Approximative Solver (Deterministic).
    [CITATION] Implements a Constructive Heuristic (Section 4.1).
    Uses 'Best-Fit' logic combined with hierarchical sorting (Ding & Chou, 2015).
    """

    # [CITATION] Ding & Chou (2015): Sorting by Discharge Port (DESC)
    load_list = sorted(containers, key=lambda c: (
        c.dischargePort, c.weight), reverse=True)

    plan: List[Tuple[Container, Slot]] = []
    left_behind: List[Container] = []

    # [CITATION] Ambrosino et al. (2004): Equilibrium Logic
    bay_order = get_center_out_order(vessel.bays)
    row_order = get_center_out_order(vessel.rows)

    # Pre-organize slots
    slots_by_bay: Dict[int, List[Slot]] = {b: [] for b in range(vessel.bays)}
    for slot in vessel.slots.values():
        slots_by_bay[slot.bay].append(slot)

    # Sort slots inside each bay list
    for b in slots_by_bay:
        slots_by_bay[b].sort(key=lambda s: (
            row_order.index(s.row),
            s.tier
        ))

    # [OPTIMIZATION] Track bay density in O(1)
    # Initialize with current state of vessel
    bay_counts: Dict[int, int] = {b: 0 for b in range(vessel.bays)}
    for s in vessel.slots.values():
        if s.container:
            bay_counts[s.bay] += 1

    for container in load_list:
        placed = False

        # Iterate Bays using Center-Out Order
        for bay_idx in bay_order:
            best_slot: Optional[Slot] = None
            best_score = float('-inf')

            # Search slots in this bay
            for slot in slots_by_bay[bay_idx]:
                if vessel.check_hard_constraints(container, slot):

                    # Pass the pre-calculated density for this bay
                    s = score_move(vessel, container, slot,
                                   bay_counts[bay_idx])

                    if s > best_score:
                        best_score = s
                        best_slot = slot

            if best_slot:
                vessel.place(container, best_slot)
                plan.append((container, best_slot))

                # [OPTIMIZATION] Update density tracker instantly
                bay_counts[best_slot.bay] += 1

                placed = True
                break

        if not placed:
            left_behind.append(container)

    return vessel, left_behind


def print_fitness_report(vessel: Vessel) -> None:
    rehandles = PhysicsUtils.calculate_rehandles(vessel)
    moments = PhysicsUtils.calculate_moments(vessel)

    print("\n--- PHYSICS & FITNESS REPORT ---")
    print(f"Total Rehandles (Overstows): {rehandles} (Lower is better)")
    print(
        f"Vertical Stability (Tier Moment): {moments['tier']:.2f} (Lower is better)")
    print(
        f"Longitudinal Balance (Bay Moment): {moments['bay']:.2f} (Target: 0.0)")
    print(
        f"Transverse Balance (Row Moment): {moments['row']:.2f} (Target: 0.0)")


def gen_case(
        bays: Range[int],
        rows: Range[int],
        tiers: Range[int],
        container_amount: Range[int],
        weight: Range[float],
        ports: Range[int],
) -> None:

    ship = Vessel(bays(), rows(), tiers())

    cargo_amt = container_amount()

    cargo = [Container.gen_random(weight, ports)
             for _ in range(cargo_amt)]

    print(
        f"\n[GENERATION] Ship Slots: {ship.capacity} | Containers: {cargo_amt}")

    ship, left_behind = heuristic_solver(cargo, ship)

    print("\n--- STOWAGE RESULTS ---")

    count = 0
    for slot in ship.slots.values():
        if slot.container:
            if count < 20:
                print(
                    f"Placed {slot.container} at Bay:{slot.bay} Row:{slot.row} Tier:{slot.tier}")
            count += 1

    if count > 20:
        print(f"... and {count-20} more.")

    if left_behind:
        print(f"\n[WARNING] Could not stow {len(left_behind)} containers.")
    else:
        print("\n[SUCCESS] All containers stowed.")

    print_fitness_report(ship)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Physics-Aware Solver with Ambrosino & Crane Balancing",
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
        print(f"Error parsing inputs: {e}")


if __name__ == "__main__":
    main()
