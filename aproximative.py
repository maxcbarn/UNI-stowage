from random import seed
from typing import List, Tuple, TypedDict

from common import Vessel, Container, Slot, SlotCoord, Range, Numeric
from itertools import product
import argparse


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

            stack = [vessel.get_slot_at(SlotCoord(b, r, t))
                     for t in range(vessel.tiers)]

            filled = [s for s in stack if s and s.container]

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


def score_move(vessel: Vessel, container: Container, slot: Slot) -> float:
    """
    Assigns a score to a placement. Higher is better.
    This integrates the MILP objectives into the greedy choice.
    """
    score = 0.0
    slot_below = None
    if slot.tier > 0:
        slot_below = vessel.get_slot_at(
            SlotCoord(slot.bay, slot.row, slot.tier - 1))

    score -= (slot.tier * container.weight) / 1000.0

    center_row = (vessel.rows - 1) / 2.0
    dist_row = abs(slot.row - center_row)
    score -= dist_row * (container.weight / 1000.0)

    if slot_below and slot_below.container:

        if container.dischargePort > slot_below.container.dischargePort:
            score -= 10000.0

        if container.weight > slot_below.container.weight:
            score -= 5000.0

    return score


def heuristic_solver(containers: List[Container], vessel: Vessel) -> Tuple[List[Tuple[Container, Slot]], List[Container]]:
    """
    Approximative Solver (Deterministic).
    Uses Best-Fit logic: scans all valid slots and picks the one with the highest physics score.
    """

    load_list = sorted(containers, key=lambda c: (
        c.dischargePort, c.weight), reverse=True)

    plan: List[Tuple[Container, Slot]] = []
    left_behind: List[Container] = []

    for container in load_list:
        best_slot = None
        best_score = float('-inf')

        for slot in vessel.slots.values():
            if vessel.check_hard_constraints(container, slot):
                s = score_move(vessel, container, slot)
                if s > best_score:
                    best_score = s
                    best_slot = slot

        if best_slot:
            vessel.place(container, best_slot)
            plan.append((container, best_slot))
        else:
            left_behind.append(container)

    return plan, left_behind


def print_fitness_report(vessel: Vessel):
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
):

    ship = Vessel(bays(), rows(), tiers())

    cargo_amt = container_amount()

    cargo = [Container.gen_random(weight, ports)
             for _ in range(cargo_amt)]

    print(
        f"\n[GENERATION] Ship Slots: {ship.capacity} | Containers: {cargo_amt}")

    plan, left_behind = heuristic_solver(cargo, ship)

    print("\n--- STOWAGE RESULTS ---")

    for i, (container, slot) in enumerate(plan):
        if i < 20:
            print(
                f"Placed {container} at Bay:{slot.bay} Row:{slot.row} Tier:{slot.tier}")
    if len(plan) > 20:
        print(f"... and {len(plan)-20} more.")

    if left_behind:
        print(f"\n[WARNING] Could not stow {len(left_behind)} containers.")
    else:
        print("\n[SUCCESS] All containers stowed.")

    print_fitness_report(ship)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Physics-Aware FFD Solver",
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
