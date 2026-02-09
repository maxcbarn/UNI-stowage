from typing import List, Dict,  Optional, TypeVar, Generic
import random
from itertools import product
from dataclasses import dataclass, field

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
        return random.random() * (self.end - self.start) + self.start


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

    def __repr__(self):
        return f"C({self.id}, P:{self.dischargePort}, {self.weight:.0f}kg)"

    @classmethod
    def genRandom(cls, weight, port, lengths, thresh):
        return cls(weight(), port(), random.choice(lengths), random.random() > thresh)


@dataclass
class Slot:
    bay: int
    row: int
    tier: int
    max_weight: float
    length: float
    reefer: bool
    container: Optional[Container] = None
    @property
    def is_free(self): return self.container is None


@dataclass(frozen=True)
class SlotCoord:
    bay: int
    row: int
    tier: int


class Vessel:
    def __init__(self, bays, rows, tiers, max_w):
        self.bays, self.rows, self.tiers = bays, rows, tiers
        self.slots: Dict[SlotCoord, Slot] = {}
        for b, r, t in product(range(bays), range(rows), range(tiers)):
            plug = (r == rows - 1)
            l = 40.0 if (b % 2 == 0) else 20.0
            self.slots[SlotCoord(b, r, t)] = Slot(b, r, t, max_w, l, plug)

    def get_slot_at(self, c): return self.slots.get(c)
    def place(self, c, s): s.container = c
    @property
    def capacity(self): return len(self.slots)

    # --- Constraints & Physics (Reused) ---
    def check_hard_constraints(self, c: Container, s: Slot):
        if not s.is_free:
            return False
        if c.weight > s.max_weight:
            return False
        if s.tier > 0:
            below = self.get_slot_at(SlotCoord(s.bay, s.row, s.tier - 1))
            if not below or below.is_free:
                return False
        if c.reefer and not s.reefer:
            return False
        if c.length != s.length:
            return False
        return True


def BuildStacks(num_bays: int, num_rows: int):
    stacks = []
    for bay in range(num_bays):
        for row in range(num_rows):
            stacks.append((bay, row))
    return stacks


def RehandlesNumber(ship: List[List[List[dict[int, float, int]]]]):
    total = 0
    for bay in ship:
        for row in bay:
            for t in range(1, len(row)):
                below = row[t - 1]
                above = row[t]

                if below is None or above is None:
                    continue

                if above['dest'] > below['dest']:
                    total += 1
    return total


def BayMoment(ship: List[List[List[dict[int, float, int]]]], baySize: int):
    center_bay = (baySize - 1) / 2.0
    total_moment = 0.0
    total_weight = 0.0

    for i, bay in enumerate(ship):
        bay_weight = 0.0
        for row in bay:
            for container in row:
                if container is not None and 'weight' in container:
                    bay_weight += container['weight']

        total_moment += bay_weight * (i - center_bay)
        total_weight += bay_weight

    if total_weight == 0:
        return 0.0

    return total_moment / total_weight


def RowMoment(ship: List[List[List[Optional[Dict[str, float]]]]], rowSize: int) -> float:
    center_row = (rowSize - 1) / 2.0
    total_moment = 0.0
    total_weight = 0.0

    for bay in ship:
        for j, row in enumerate(bay):
            row_weight = 0.0

            for container in row:
                if container is not None and 'weight' in container:
                    row_weight += container['weight']

            total_moment += row_weight * (j - center_row)
            total_weight += row_weight

    if total_weight == 0:
        return 0.0

    return total_moment / total_weight


def TierMoment(ship: List[List[List[Optional[Dict[str, float]]]]]) -> float:
    total_moment = 0.0
    total_weight = 0.0

    for bay in ship:
        for row in bay:
            for t, container in enumerate(row):
                if container is not None and 'weight' in container:
                    weight = container['weight']

                    total_moment += weight * t
                    total_weight += weight

    if total_weight == 0:
        return 0.0

    return total_moment / total_weight


def ContainerRamdom(number: int) -> dict[int, float, int]:

    # Generating 250 containers starting from id 11
    containers_250 = [
        {
            'id': i,
            'weight': round(random.uniform(1.0, 100.0), 1),
            'dest': random.randint(1, 5)
        }
        for i in range(1, number + 1)
    ]

    return containers_250


def calculate_cost(vessel: Vessel, leftovers: List[Container]) -> float:
    # 1. Rehandles
    rehandles = 0
    for b, r in product(range(vessel.bays), range(vessel.rows)):
        stack = [vessel.get_slot_at(SlotCoord(b, r, t))
                 for t in range(vessel.tiers)]
        filled = [s.container for s in stack if s.container]
        for i in range(1, len(filled)):
            if filled[i].dischargePort > filled[i-1].dischargePort:
                rehandles += 1

    # 2. Moments
    # (Simplified for MCTS speed: just track Stability)
    tier_moment = sum(
        s.tier * s.container.weight for s in vessel.slots.values() if s.container)

    # Cost Function
    cost = (rehandles * 1000.0) + (tier_moment * 0.1) + \
        (len(leftovers) * 5000.0)
    return cost
