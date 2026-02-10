from typing import List, Dict,  Optional, Tuple, TypeVar, Generic, TypedDict
import random
from itertools import product
from dataclasses import dataclass, field

Numeric = TypeVar("Numeric", int, float)


class Cont(TypedDict):
    id: int
    weight: float
    dest: int


type List3D[T] = List[List[List[T]]]
type Ship = List3D[Optional[Cont]]


@dataclass
class CostReport:
    ship: Ship
    rehandles: int = field(init=False)
    moments: Tuple[float, float, float] = field(init=False)

    def __post_init__(self):
        self.rehandles = RehandlesNumber(self.ship)
        self.moments = BayMoment(self.ship), RowMoment(
            self.ship), TierMoment(self.ship)

    @property
    def bay_moment(self) -> float:
        return BayMoment(self.ship)

    @property
    def row_moment(self) -> float:
        return RowMoment(self.ship)

    @property
    def tier_moment(self) -> float:
        return TierMoment(self.ship)


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
    _static_id: int = field(default=0, repr=False)

    def __post_init__(self):
        type(self)._static_id += 1
        self.id = type(self)._static_id

    def __repr__(self):
        return f"C({self.id}, P:{self.dischargePort}, {self.weight:.0f}kg)"

    @classmethod
    def from_cont(cls, cont: Cont):
        return cls(cont['weight'], cont['dest'])

    @classmethod
    def gen_random(cls, weight: Range[float], port: Range[int]):
        return cls(weight(), port())


@dataclass
class Slot:
    bay: int
    row: int
    tier: int
    container: Optional[Container] = None
    @property
    def is_free(self): return self.container is None


@dataclass(frozen=True)
class SlotCoord:
    bay: int
    row: int
    tier: int


class Vessel:
    def __init__(self, bays: int = 0, rows: int = 0, tiers: int = 0):
        self.bays, self.rows, self.tiers = bays, rows, tiers
        self.slots: Dict[SlotCoord, Slot] = {}
        for b, r, t in product(range(bays), range(rows), range(tiers)):
            self.slots[SlotCoord(b, r, t)] = Slot(b, r, t)

    @classmethod
    def from_ship(cls, ship: Ship):
        vessel = cls()

        num_bays = len(ship)
        num_rows = len(ship[0]) if num_bays > 0 else 0
        num_tiers = len(ship[0][0]) if num_rows > 0 else 0

        vessel.bays = num_bays
        vessel.rows = num_rows
        vessel.tiers = num_tiers

        for b, bay in enumerate(ship):
            for r, row in enumerate(bay):
                for t, tier in enumerate(row):
                    coord = SlotCoord(b, r, t)
                    vessel.slots[coord] = Slot(b, r, t)
                    if tier:
                        vessel.slots[coord].container = Container.from_cont(
                            tier)

        return vessel

    def get_slot_at(self, c: SlotCoord): return self.slots.get(c)

    def place(self, c: Container, s: Slot): s.container = c
    @property
    def capacity(self): return len(self.slots)

    # --- Constraints & Physics (Reused) ---
    def check_hard_constraints(self, c: Container, s: Slot):
        if not s.is_free:
            return False
        if s.tier > 0:
            below = self.get_slot_at(SlotCoord(s.bay, s.row, s.tier - 1))
            if not below or below.is_free:
                return False
        return True


def BuildStacks(num_bays: int, num_rows: int):
    stacks: List[Tuple[int, int]] = []
    for bay in range(num_bays):
        for row in range(num_rows):
            stacks.append((bay, row))
    return stacks


def RehandlesNumber(ship: Ship) -> int:
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


def BayMoment(ship: Ship):
    baySize = len(ship)
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


def RowMoment(ship: Ship) -> float:
    rowSize = len(ship[0])
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


def TierMoment(ship: Ship) -> float:
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


def ContainerRamdom(number: int) -> List[Cont]:

    # Generating 250 containers starting from id 11
    containers_250: List[Cont] = [
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

        def get(s: Slot | None): return s.container if s else None
        filled: List[Optional[Container]] = [get(s) for s in stack if get(s)]
        for i in range(1, len(filled)):
            curr = filled[i]
            prev = filled[i-1]
            if curr and prev and curr.dischargePort > prev.dischargePort:
                rehandles += 1

    # 2. Moments
    # (Simplified for MCTS speed: just track Stability)
    tier_moment = sum(
        s.tier * s.container.weight for s in vessel.slots.values() if s.container)

    # Cost Function
    cost = (rehandles * 1000.0) + (tier_moment * 0.1) + \
        (len(leftovers) * 5000.0)

    return cost


def container_to_cont(c: Container | None) -> Optional[Cont]:
    if not c:
        return None

    return {
        'id': c.id,
        'weight': c.weight,
        'dest': c.dischargePort
    }


def vessel_to_ship(vessel: Vessel) -> Ship:
    def getContainer(s: Slot | None):
        if s is None:
            return None
        return s.container

    ship: Ship = [[[container_to_cont(getContainer(vessel.get_slot_at(SlotCoord(b, r, t)))) for t in range(
        vessel.tiers)] for r in range(vessel.rows)] for b in range(vessel.bays)]

    return ship


def ship_to_vessel(ship: Ship) -> Tuple[Vessel, List[Container]]:
    return Vessel.from_ship(ship), []
