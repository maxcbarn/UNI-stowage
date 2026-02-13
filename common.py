from __future__ import annotations
import json

import numpy as np
from typing import List, Dict, Optional, Tuple, TypeVar, Generic, TypedDict
import random
from itertools import product
from dataclasses import dataclass, field
from pathlib import Path


# Sum of 'constWeight' for all 24 bays (Calculated from file)
LIGHTSHIP_WEIGHT = 60787.0

# Weighted Average of 'constWeighVcg' (All bays are 18.0 in your file)
LIGHTSHIP_VCG = 18.0

# Hydrostatic Table: Displacement (tons) -> KM (Metacenter Height in meters)
HYDRO_X = [
    54037.0, 67172.0, 80797.0, 94889.0, 109446.0, 124451.0, 139900.0,
    155794.0, 172144.0, 188981.0, 206337.0, 224263.0, 242694.0, 261543.0,
    280761.0, 300312.0, 320171.0, 340320.0, 360740.0, 381416.0, 402332.0,
    423469.0, 444807.0, 465746.0, 486668.0, 507733.0, 511962.0
]

HYDRO_Y = [
    45.70, 40.72, 36.84, 34.14, 32.21, 30.67, 29.50,
    28.61, 27.94, 27.45, 27.11, 26.90, 26.77, 26.68,
    26.64, 26.63, 26.66, 26.71, 26.79, 26.90, 27.03,
    27.17, 27.32, 26.92, 27.10, 27.30, 27.34
]

# --- PHYSICS CONSTANTS (The Source of Truth) ---
# Meters (Vertical Center of Gravity of empty ship)
AVG_CONTAINER_WEIGHT = 14.0  # Tons (Fallback)
CONTAINER_HEIGHT = 2.6       # Meters
MIN_GM = 1.0                 # Meters (Minimum safety limit)

# --- UNIFIED WEIGHTS ---
W_REHANDLE = 10000.0    # Efficiency
W_GM_FAIL = 50000.0     # Safety (Highest Priority)
W_BALANCE = 1.0         # Quality (Tie-breaker)
W_LEFTOVER = 100000.0   # Critical (Must load cargo)

Numeric = TypeVar("Numeric", int, float)


class Cont(TypedDict):
    id: int
    weight: float
    dest: int


type List3D[T] = List[List[List[T]]]
type Ship = List3D[Optional[Cont]]


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
    id: int
    weight: float
    height: float
    dischargePort: int

    def __repr__(self):
        return f"C({self.id}, P:{self.dischargePort}, {self.weight:.0f}kg)"


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


@dataclass
class Vessel:
    bays: int = 0
    rows: int = 0
    tiers: int = 0

    hydro_disp: List[float] = field(default_factory=lambda: HYDRO_X)
    hydro_km: List[float] = field(default_factory=lambda: HYDRO_Y)

    bays: int = 0
    rows: int = 0
    tiers: int = 0
    lightship_weight: float = LIGHTSHIP_WEIGHT
    lightship_vcg: float = LIGHTSHIP_VCG

    def __post_init__(self):
        self.slots: Dict[SlotCoord, Slot] = {}
        for b, r, t in product(range(self.bays), range(self.rows), range(self.tiers)):
            self.slots[SlotCoord(b, r, t)] = Slot(b, r, t)

    def JSON_str(self) -> str:
        return json.dumps({
            "contCount": self.containerAmount,
            "bays": self.bays,
            "rows": self.rows,
            "tiers": self.tiers,
            "lightship_weight": self.lightship_weight,
            "lightship_vcg": self.lightship_vcg,
            "slots": [
                {
                    "bay": coord.bay,
                    "row": coord.row,
                    "tier": coord.tier,
                    "container": {
                        "id": slot.container.id,
                        "weight": slot.container.weight,
                        "height": slot.container.height,
                        "dischargePort": slot.container.dischargePort,
                    } if slot.container else None,
                }
                for coord, slot in self.slots.items()
                if slot.container
            ],
        })

    @property
    def containerAmount(self):
        return sum(1 for s in self.slots.values() if s.container)

    @property
    def capacity(self): return len(self.slots)

    def get_slot_at(self, c: SlotCoord): return self.slots.get(c)
    def place(self, c: Container, s: Slot): s.container = c

    def check_hard_constraints(self, c: Container, s: Slot):
        if not s.is_free:
            return False
        if s.tier > 0:
            below = self.get_slot_at(SlotCoord(s.bay, s.row, s.tier - 1))
            if not below or below.is_free:
                return False
        return True

    def calculate_rehandles(self) -> int:
        """
        Counts overstows: heavy/later-port containers on top of light/earlier-port ones.
        Optimized to iterate directly over slots without converting to 3D list.
        """
        total = 0
        # Iterate only filled slots above tier 0
        for coord, slot in self.slots.items():
            if slot.tier > 0 and slot.container:
                # Look strictly one tier down
                below_coord = SlotCoord(coord.bay, coord.row, coord.tier - 1)
                below_slot = self.slots.get(below_coord)

                if below_slot and below_slot.container:
                    # Check Discharge Port Inversion (Heuristic standard)
                    if slot.container.dischargePort > below_slot.container.dischargePort:
                        total += 1
        return total

    def calculate_bay_moment(self) -> float:
        """Calculates Longitudinal Moment (Trim)."""
        if self.containerAmount == 0:
            return 0.0

        center_bay = (self.bays - 1) / 2.0
        total_moment = 0.0
        total_weight = 0.0

        for slot in self.slots.values():
            if slot.container:
                w = slot.container.weight
                total_weight += w
                total_moment += w * (slot.bay - center_bay)

        return total_moment / total_weight if total_weight > 0 else 0.0

    def calculate_row_moment(self) -> float:
        """Calculates Transverse Moment (List/Heel)."""
        if self.containerAmount == 0:
            return 0.0

        center_row = (self.rows - 1) / 2.0
        total_moment = 0.0
        total_weight = 0.0

        for slot in self.slots.values():
            if slot.container:
                w = slot.container.weight
                total_weight += w
                total_moment += w * (slot.row - center_row)

        return total_moment / total_weight if total_weight > 0 else 0.0

    def calculate_tier_moment(self) -> float:
        """Calculates Vertical Moment (Proxy for VCG)."""
        if self.containerAmount == 0:
            return 0.0

        total_moment = 0.0
        total_weight = 0.0

        for slot in self.slots.values():
            if slot.container:
                w = slot.container.weight
                total_weight += w
                total_moment += w * slot.tier

        return total_moment / total_weight if total_weight > 0 else 0.0


def calculate_gm(vessel: Vessel) -> float:
    """Calculates GM from either a Vessel object or a Ship (List3D)."""
    cargo_weight = 0.0
    cargo_moment = 0.0

    slots = vessel.slots.values()
    for slot in slots:
        if slot.container:
            w = slot.container.weight
            h = vessel.lightship_vcg + (slot.tier * CONTAINER_HEIGHT)
            cargo_weight += w
            cargo_moment += (w * h)

    disp = vessel.lightship_weight + cargo_weight
    if disp == 0:
        return 20.0  # Fallback for empty/error

    vcg = ((vessel.lightship_weight * vessel.lightship_vcg) + cargo_moment) / disp
    km = np.interp(disp, vessel.hydro_disp, vessel.hydro_km)
    return km - vcg


def calculate_cost(vessel: Vessel, leftovers: List[Container] = []) -> float:
    # 1. Safety (GM)
    # Note: calculate_gm logic is also partly redundant now,
    # but we keep it for the hydrostatic lookup.
    gm = calculate_gm(vessel)
    cost_gm = (MIN_GM - gm) * W_GM_FAIL if gm < MIN_GM else 0.0

    # 2. Efficiency (Rehandles) - DIRECT CALL
    cost_rehandles = vessel.calculate_rehandles() * W_REHANDLE

    # 3. Balance (Moments) - DIRECT CALLS
    cost_balance = (abs(vessel.calculate_bay_moment()) +
                    abs(vessel.calculate_row_moment())) * W_BALANCE

    # 4. Critical (Leftovers)
    cost_leftover = len(leftovers) * W_LEFTOVER

    return cost_gm + cost_rehandles + cost_balance + cost_leftover

# --- DATA STRUCTURES & UTILS ---


@dataclass
class CostReport:
    vessel: Vessel
    rehandles: int = field(init=False)
    moments: Tuple[float, float, float] = field(init=False)
    gm: float = field(init=False)
    total_cost: float = field(init=False)

    def __post_init__(self):
        self.rehandles = self.vessel.calculate_rehandles()
        self.moments = (self.vessel.calculate_bay_moment(
        ), self.vessel.calculate_row_moment(), self.vessel.calculate_tier_moment())
        self.gm = calculate_gm(self.vessel)

        # Calculate cost parts
        c_gm = (MIN_GM - self.gm) * W_GM_FAIL if self.gm < MIN_GM else 0.0
        c_re = self.rehandles * W_REHANDLE
        c_bal = (abs(self.moments[0]) + abs(self.moments[1])) * W_BALANCE
        self.total_cost = c_gm + c_re + c_bal

    @classmethod
    def header(cls):
        return "bays, rows, tiers, lightVCG, lightWeight, hydroDisp(x), hydroKM(y),cost, rehandles, bayMoment, rowMoment, tierMoment, gm".replace(' ', '')

    def log(self):
        return f'{self.vessel.bays}, {self.vessel.rows}, {self.vessel.tiers}, {self.vessel.lightship_vcg}, {self.vessel.lightship_weight}, "{self.vessel.hydro_km}", "{self.vessel.hydro_disp}", {self.total_cost}, {self.rehandles}, {self.bay_moment}, {self.row_moment}, {self.tier_moment}, {self.gm}'.replace(" ", '')

    @property
    def bay_moment(self) -> float: return self.moments[0]
    @property
    def row_moment(self) -> float: return self.moments[1]
    @property
    def tier_moment(self) -> float: return self.moments[2]


def conts_to_containers(cs: List[Cont]) -> List[Container]:
    return [Container(c['id'], c['weight'], 2.745, c['dest']) for c in cs]


def parse_benchmark_vessel(filepath: Path) -> Vessel:
    """
    Parses vessel_L.txt to extract dimensions AND physics data.

    Extracts:
    1. Global Dimensions (# Ship)
    2. Hydrostatic Table (## HydroPoints -> Displacement, Metacenter)
    3. Lightship Weight & VCG (Sum/Avg of ## Bay -> constWeight, constWeighVcg)
    """
    # Default values
    bays, rows, tiers = 0, 0, 0
    hydro_disp: List[float] = []
    hydro_km: List[float] = []

    total_bay_weight = 0.0
    total_bay_moment = 0.0

    # State flags
    section_hydro = False

    with open(filepath, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # --- 1. Dimensions ---
        if line.startswith("# Ship:"):
            # Format: # Ship: bays stacks tiers ...
            # Next line: 24 22 21 0.100
            parts = lines[i+1].split()
            bays = int(parts[0])
            rows = int(parts[1])
            tiers = int(parts[2])
            section_hydro = False

        # --- 2. Hydrostatics ---
        elif line.startswith("## HydroPoints:"):
            # Format: displacement minLcg maxLcg metacenter
            section_hydro = True
            continue  # Skip header

        elif line.startswith("##"):
            # Any other section header kills the Hydro state
            section_hydro = False

        # --- 3. Lightship Weight (Bay Sections) ---
        if line.startswith("## Bay:"):
            # Format: ## Bay: index lcg ... constWeight constWeighVcg
            # Next line: 0 161.2 -2450 1850 46750 2440.0 18.0
            data_line = lines[i+1].strip()
            parts = data_line.split()

            # Index 5 = constWeight, Index 6 = constWeighVcg
            if len(parts) >= 7:
                w = float(parts[5])
                vcg = float(parts[6])

                total_bay_weight += w
                total_bay_moment += (w * vcg)

        # --- Data Parsing (Context Sensitive) ---
        elif section_hydro:
            # We are inside the HydroPoints block
            parts = line.split()
            if len(parts) == 4 and parts[0].replace('.', '', 1).isdigit():
                # Col 0: Displacement, Col 3: Metacenter (KM)
                hydro_disp.append(float(parts[0]))
                hydro_km.append(float(parts[3]))

    # Calculate Weighted Average VCG for the Lightship
    if total_bay_weight > 0:
        lightship_vcg = total_bay_moment / total_bay_weight
    else:
        lightship_vcg = 0.0

    return Vessel(
        bays=bays,
        rows=rows,
        tiers=tiers,
        lightship_weight=total_bay_weight,
        lightship_vcg=lightship_vcg,
        hydro_disp=hydro_disp,
        hydro_km=hydro_km
    )


def parse_benchmark_containers(filepath: Path) -> List[Container]:
    """
    Parses VLHigh1.txt for container weights, heights, and load list.

    Transport type table format:
        id  length=(20|40)  weight  type=(DC|RC|HC|HR)

    Height is derived from the type column:
        HC / HR  →  2.9 m  (High Cube)
        DC / RC  →  2.6 m  (Standard)
    """
    # Standard container height (DC, RC) in metres
    CONTAINER_HEIGHT_STANDARD = 2.6
    # High-Cube container height (HC, HR) in metres
    CONTAINER_HEIGHT_HC = 2.9

    # Container types that are High Cube
    _HIGH_CUBE_TYPES = {"HC", "HR"}

    containers: List[Container] = []

    # Maps transport type id → (weight, height)
    transport_props: Dict[int, tuple[float, float]] = {}

    with open(filepath, "r") as f:
        lines = f.readlines()

    mode = None

    for line in lines:
        line = line.strip()
        if not line or line.startswith("//"):
            continue

        # --- Section detection ---
        if "# Transport type" in line:
            mode = "WEIGHTS"
            continue
        elif "# Container" in line:   # matches 'Container' and 'Containers'
            mode = "LOADLIST"
            continue

        parts = line.split()

        # Skip section headers and comment lines (first token not a digit)
        if not parts[0].isdigit():
            continue

        try:
            if mode == "WEIGHTS":
                # Format: id  length  weight  type
                # e.g.:   21  40      3       HC
                t_id = int(parts[0])
                weight = float(parts[2])
                container_type = parts[3].upper()
                height = (
                    CONTAINER_HEIGHT_HC
                    if container_type in _HIGH_CUBE_TYPES
                    else CONTAINER_HEIGHT_STANDARD
                )
                transport_props[t_id] = (weight, height)

            elif mode == "LOADLIST":
                # Format (with position):   startPort endPort typeId bay stack tier slot
                # Format (without position): startPort endPort typeId
                dest_port = int(parts[1])
                type_id = int(parts[2])

                if type_id in transport_props:
                    weight, height = transport_props[type_id]
                    containers.append(
                        Container(
                            id=len(containers),
                            weight=weight,
                            height=height,
                            dischargePort=dest_port,
                        )
                    )

        except (ValueError, IndexError):
            continue

    return containers
