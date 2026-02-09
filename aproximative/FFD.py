from collections import defaultdict
from dataclasses import dataclass, field
from random import choice, randrange, random, seed
from typing import DefaultDict, Dict, List, Tuple, TypeVar, Generic
from itertools import product
import argparse


Numeric = TypeVar("Numeric", int, float)


@dataclass
class Range(Generic[Numeric]):
    start: Numeric
    end: Numeric

    def __call__(self) -> Numeric:
        if isinstance(self.start, int) and isinstance(self.end, int):
            return randrange(self.start, self.end)
        else:
            return random() * (self.end - self.start) + self.start


type ChoiceDomain[T] = list[T]


@dataclass
class Container:
    id: int = field(init=False)
    weight: float
    dischargePort: int
    length: float
    reefer: bool

    _static_id: int = field(default=0, repr=False)

    def __post_init__(self):
        type(self)._static_id += 1
        self.id = type(self)._static_id

    @classmethod
    def genRandom(cls, weight: Range[float], port: Range[int], length_domain: ChoiceDomain[float], threshold: float):
        return cls(weight(), port(), choice(length_domain), random() > threshold)


@dataclass
class Slot:
    bay: int
    row: int
    tier: int
    max_weight: float
    length: float
    reefer: bool
    container: Container | None = None

    @property
    def is_free(self):
        return self.container is None


@dataclass(frozen=True)
class SlotCoord:
    bay: int
    row: int
    tier: int


@dataclass(init=False)
class Vessel:
    bays: int
    rows: int
    tiers: int
    max_weight: float = 30.0
    slots: Dict[SlotCoord, Slot] = field(
        default_factory=lambda: dict(),
        init=False,
        repr=False
    )

    @property
    def capacity(self):
        return len(self.slots)

    def __init__(self, bays: int, rows: int, tiers: int, weight: Range[float], lenght_domain: ChoiceDomain = [20, 40], threshold: float = 0.5):
        self.bays = bays
        self.rows = rows
        self.tiers = tiers
        self.slots = {}

        for b, r, t in product(range(self.bays), range(self.rows), range(self.tiers)):
            self.slots[SlotCoord(b, r, t)] = Slot(
                b, r, t, weight(), choice(lenght_domain), random() > threshold)

    def get_first_valid_slot(self, container: Container):
        """
        The Core FFD Logic:
        Scan slots in a fixed order and find the FIRST one that fits.
        """
        for b, r, t in product(range(self.bays), range(self.rows), range(self.tiers)):
            slot = self.get_slot_at(SlotCoord(b, r, t))
            if slot and self.check_hard_constraints(container, slot):
                return slot
        return None

    def check_hard_constraints(self, container: Container, slot: Slot) -> bool:

        if not slot.is_free:
            return False

        if container.weight > slot.max_weight:
            return False

        if slot.tier > 0:
            slot_below = self.get_slot_at(
                SlotCoord(slot.bay, slot.row, slot.tier - 1)
            )
            if slot_below and slot_below.is_free:
                return False

        if container.reefer and not slot.reefer:
            return False

        if container.length != slot.length:
            return False
        return True

    def get_slot_at(self, coord: SlotCoord) -> Slot | None:
        return self.slots.get(coord)

    def place(self, container: Container, slot: Slot) -> None:
        slot.container = container


def first_fit_decreasing_solver(containers: List[Container], vessel: Vessel) -> Tuple[List[Tuple[Container, Slot]], List[Container]]:
    """
    Implementation of the FFD heuristic described in the report.
    """

    sorted_containers = sorted(
        containers,
        key=lambda c: (c.dischargePort, c.weight),
        reverse=True
    )

    print(f"Vessel: {vessel}")
    print(
        f"Sorted Load List ({len(sorted_containers)}): ", *sorted_containers, sep='\n\t', end='\n\n')

    stowage_plan: List[Tuple[Container, Slot]] = []
    unloaded_containers: List[Container] = []

    for container in sorted_containers:

        target_slot = vessel.get_first_valid_slot(container)

        if target_slot:
            vessel.place(container, target_slot)
            stowage_plan.append((container, target_slot))
        else:

            unloaded_containers.append(container)

    return stowage_plan, unloaded_containers


@dataclass
class PenaltyWeights:
    overstow: float
    stability: float


@dataclass
class FitnessReport:
    total_cost: float
    overstow_count: int
    stability_inversions: int


def calculate_fitness(vessel: Vessel, penalty_weights: PenaltyWeights) -> FitnessReport:
    """
    Calculates the 'Fitness' (Total Cost) of a stowage plan based on
    the heuristic scoring rules defined in the syllabus plan.
    Lower Cost = Better Fitness.
    """
    total_cost = 0
    overstow_count = 0
    stability_inversions = 0

    stacks: Dict[Tuple[int, int], List[Slot]] = {}
    for slot in vessel.slots.values():
        if slot.container:
            key = (slot.bay, slot.row)
            if key not in stacks:
                stacks[key] = []
            stacks[key].append(slot)

    for stack_key, slots in stacks.items():

        slots.sort(key=lambda s: s.tier)

        for i in range(len(slots) - 1):
            lower_slot = slots[i]
            upper_slot = slots[i+1]

            lower_c = lower_slot.container
            upper_c = upper_slot.container
            if not lower_c or not upper_c:
                print(lower_c, upper_c)
                continue

            if upper_c.dischargePort > lower_c.dischargePort:
                overstow_count += 1
                total_cost += penalty_weights.overstow
                print(
                    f"  [Penalty] Overstow at {stack_key}: {upper_c} blocks {lower_c}")

            if upper_c.weight > lower_c.weight:
                stability_inversions += 1
                total_cost += penalty_weights.stability
                print(
                    f"  [Penalty] Unstable Stack at {stack_key}: {upper_c} crushes {lower_c}")

    return FitnessReport(total_cost, overstow_count, stability_inversions)


def gen_case(
        bays: Range[int],
        rows: Range[int],
        tiers: Range[int],
        container_amount: Range[int],
        weight: Range[float],
        ports: Range[int],
        fitness_metric: PenaltyWeights
):
    ship = Vessel(bays(), rows(), tiers(),
                  Range[float](weight.end, weight.end + 1))

    cargo_amount = container_amount()
    cargo = [Container.genRandom(weight, ports, [20, 40], 0.8)
             for _ in range(cargo_amount)]

    plan, left_behind = first_fit_decreasing_solver(cargo, ship)

    print("\nStowage Results:")
    for container, slot in plan:
        print(
            f"Placed {container} at Bay:{slot.bay} Row:{slot.row} Tier:{slot.tier}")

    if left_behind:
        print(
            f"\nWARNING: Could not stow {len(left_behind)} containers: {left_behind}")

    print(calculate_fitness(ship, fitness_metric))


def exemple():

    ship = Vessel(2, 2, 3, Range[float](1_000, 10_000))

    cargo = [
        Container(10, 1, 20, False),
        Container(25, 3, 20, False),
        Container(15, 2, 20, False),
        Container(5,  1, 20, False),
        Container(20, 3, 20, False),
    ]

    plan, left_behind = first_fit_decreasing_solver(cargo, ship)

    print("\nStowage Results:")
    for container, slot in plan:
        print(
            f"Placed {container} at Bay:{slot.bay} Row:{slot.row} Tier:{slot.tier}")

    if left_behind:
        print(
            f"\nWARNING: Could not stow {len(left_behind)} containers: ", *left_behind, sep='\n\t', end='\n\n')

    print(calculate_fitness(ship, PenaltyWeights(1000, 100)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FDD Stowage Solver CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    ship_group = parser.add_argument_group("Ship Configuration")
    ship_group.add_argument("--bays", nargs="+", type=int, default=[5],
                            help="Number of Bays. Provide 1 value for exact, 2 for range (min max).")
    ship_group.add_argument("--rows", nargs="+", type=int, default=[5],
                            help="Number of Rows. Provide 1 value for exact, 2 for range (min max).")
    ship_group.add_argument("--tiers", nargs="+", type=int, default=[5],
                            help="Number of Tiers. Provide 1 value for exact, 2 for range (min max).")

    cargo_group = parser.add_argument_group("Cargo Configuration")
    cargo_group.add_argument("--containers", nargs="+", type=int, default=[50, 100],
                             help="Number of Containers. Provide 1 value for exact, 2 for range (min max).")
    cargo_group.add_argument("--weight", nargs="+", type=float, default=[1000.0, 30000.0],
                             help="Container Weight (kg). Provide 1 value for exact, 2 for range (min max).")
    cargo_group.add_argument("--ports", nargs="+", type=int, default=[1, 5],
                             help="Discharge Ports (e.g. 1 5). Provide 1 value for exact, 2 for range (min max).")

    heuristic_group = parser.add_argument_group("Heuristic Parameters")
    heuristic_group.add_argument("--penalty-overstow", type=float, default=1000.0,
                                 help="Cost penalty for each overstow event.")
    heuristic_group.add_argument("--penalty-stability", type=float, default=100.0,
                                 help="Cost penalty for each weight inversion.")

    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility. If None, uses system time.")

    args = parser.parse_args()

    def to_range(values: List[Numeric], is_float: bool = False) -> Range:
        if len(values) == 1:
            start = values[0]
            if is_float:
                return Range(float(start), float(start))
            else:
                return Range(int(start), int(start) + 1)

        elif len(values) == 2:
            return Range(values[0], values[1])
        else:
            raise ValueError("Arguments must be 1 or 2 values (min [max])")

    if args.seed is not None:
        seed(args.seed)
        print(f"--- Running with SEED: {args.seed} ---")
    else:
        print(f"--- Running with Random Seed (System Time) ---")

    try:
        print(f"--- FFD Heuristic Simulation ---")

        penalties = PenaltyWeights(
            args.penalty_overstow, args.penalty_stability)

        gen_case(
            bays=to_range(args.bays),
            rows=to_range(args.rows),
            tiers=to_range(args.tiers),
            container_amount=to_range(args.containers),
            weight=to_range(args.weight, is_float=True),
            ports=to_range(args.ports),
            fitness_metric=penalties
        )

    except ValueError as e:
        print(f"Error parsing ranges: {e}")


if __name__ == "__main__":
    main()
