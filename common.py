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
AVG_CONTAINER_WEIGHT = 14.0  # Tons (Fallback)
CONTAINER_HEIGHT = 2.6       # Meters (Standard container height — used as fallback only)
MIN_GM = 1.0                 # Meters (Minimum safety limit)

# FIX 1: Added DECK_HEIGHT as the true vertical reference for cargo.
# This is the height of the weather deck above the keel (baseline).
# Cargo VCG is measured from the keel, so tier-0 containers sit at DECK_HEIGHT,
# NOT at lightship_vcg (which is the hull's center of gravity, not the deck).
# Set this to the actual vessel deck height in metres. 13.0m is a typical value
# for a large container ship; override per vessel if you have exact data.
DECK_HEIGHT = 13.0  # Meters (keel to weather deck)

# --- UNIFIED WEIGHTS ---
W_REHANDLE = 10000.0    # Efficiency
W_GM_FAIL = 50000.0     # Safety (Highest Priority)
W_BALANCE = 1.0         # Quality (Tie-breaker)
W_LEFTOVER = 100000.0   # Critical (Must load cargo)

W_STACK_BONUS = 5_000.0     # Bonus for matching discharge ports (Homogeneous stack)
W_CRANE_BALANCE = 500.0     # Penalty for overloading one bay (Parallel ops)

# Stability Weights
W_WEIGHT_INVERSION = 5_000.0  # Penalty for Heavy on Light
# FIX 2: Recalibrated moment weights so stability competes with other terms.
# Previously 1.0 each, which made them ~3000x weaker than W_CRANE_BALANCE (500)
# on a typical 30-ton container at tier 3 (score delta = 0.09 vs 500+).
# New values bring stability penalties into the same order of magnitude as
# W_WEIGHT_INVERSION and W_STACK_BONUS (both 5000).
W_MOMENT_TIER = 200.0   # Minimize VCG (Vertical)   — was 1.0
W_MOMENT_ROW  = 300.0   # Minimize List (Transverse) — was 1.0

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
    vcg: float = 0.0
    container: Optional[Container] = None

    @property
    def is_free(self): return self.container is None


@dataclass(frozen=True)
class SlotCoord:
    bay: int
    row: int
    tier: int


# FIX 3: Removed duplicate field declarations for bays, rows, tiers.
# The original dataclass declared each of them twice; Python silently used
# the last declaration, but it was a latent bug waiting to cause trouble
# during any refactor. Now each field appears exactly once, in logical order.
@dataclass
class Vessel:
    bays: int = 0
    rows: int = 0
    tiers: int = 0
    lightship_weight: float = LIGHTSHIP_WEIGHT
    lightship_vcg: float = LIGHTSHIP_VCG
    hydro_disp: List[float] = field(default_factory=lambda: HYDRO_X)
    hydro_km: List[float] = field(default_factory=lambda: HYDRO_Y)
    # deck_height: height of weather deck above keel (metres).
    # Cargo VCG is referenced from the keel, so tier-0 containers sit at deck_height.
    deck_height: float = DECK_HEIGHT

    def __post_init__(self):
        self.slots: Dict[SlotCoord, Slot] = {}
        for b, r, t in product(range(self.bays), range(self.rows), range(self.tiers)):
            self.slots[SlotCoord(b, r, t)] = Slot(b, r, t)

    def clear(self):
        for k in self.slots.keys():
            self.slots[k].container = None

    def JSON_str(self) -> str:
        rep = CostReport(self)
        return json.dumps({
            # --- State Metadata ---
            "contCount": self.containerAmount,
            "bays": self.bays,
            "rows": self.rows,
            "tiers": self.tiers,
            
            # --- Physical Constants ---
            "lightship_weight": self.lightship_weight,
            "lightship_vcg": self.lightship_vcg,
            "deck_height": self.deck_height,
            
            # --- Hydrostatic Tables (Crucial for validation) ---
            "hydro_disp": self.hydro_disp,
            "hydro_km": self.hydro_km,
            
            # --- Calculated Metrics ---
            "cost": rep.total_cost,
            "rehandles": rep.rehandles,
            "gm": rep.gm,
            "rowMoment": rep.row_moment,
            "bayMoment": rep.bay_moment,
            "tierMoment": rep.tier_moment,
            
            # --- Slot/Cargo Data ---
            "slots": [
                {
                    "bay": coord.bay,
                    "row": coord.row,
                    "tier": coord.tier,
                    "vcg": slot.vcg, # Critical: The stack base height
                    "container": {
                        "id": slot.container.id,
                        "weight": slot.container.weight,
                        "height": slot.container.height,
                        "dischargePort": slot.container.dischargePort,
                    } if slot.container else None
                }
                for coord, slot in self.slots.items()
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
        Counts overstows: later-port containers stacked on top of earlier-port ones.
        Iterates directly over slots without converting to a 3D list.
        """
        total = 0
        for coord, slot in self.slots.items():
            if slot.tier > 0 and slot.container:
                below_coord = SlotCoord(coord.bay, coord.row, coord.tier - 1)
                below_slot = self.slots.get(below_coord)
                if below_slot and below_slot.container:
                    if slot.container.dischargePort > below_slot.container.dischargePort:
                        total += 1
        return total

    def calculate_moments(self) -> Tuple[float, float, float]:
        return self.calculate_bay_moment(), self.calculate_row_moment(), self.calculate_tier_moment()

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
    """
    Calculates GM (metacentric height) with corrected vertical positioning.

    FIX:
    - Distinguishes between HOLD (Tier < 80) and DECK (Tier >= 80) containers.
    - HOLD containers start stacking from the tank top (approx. 0.0m).
    - DECK containers start stacking from vessel.deck_height.
    - This avoids the "flat barge" error where hold cargo was artificially raised.
    """
    
    # --- Constants for ISO Tier Heuristic ---
    TIER_ON_DECK_START = 80   # ISO standard: 82 is usually the first deck tier
    TANK_TOP_HEIGHT = 0.5     # Approx height of hold bottom above keel (adjust if needed)
    
    cargo_weight = 0.0
    cargo_moment = 0.0

    # Group slots by stack (bay, row) to process them bottom-up
    stacks: Dict[Tuple[int, int], List[Slot]] = {}
    for slot in vessel.slots.values():
        if slot.container:
            key = (slot.bay, slot.row)
            stacks.setdefault(key, []).append(slot)

    for stack_slots in stacks.values():
        # Sort by tier to stack from bottom to top
        stack_slots.sort(key=lambda s: s.tier)
        
        # Track the top of the 'current' stack. 
        # We initialize at tank top, but will jump if we hit a deck tier.
        current_base_height = TANK_TOP_HEIGHT
        
        for slot in stack_slots:
            # CHECK: Are we transitioning to the deck?
            if slot.tier >= TIER_ON_DECK_START:
                # If we are on deck, the base of this container MUST be at least at deck_height.
                # We use max() to ensure we don't drop down if the hold stack 
                # somehow exceeds deck height (unlikely but safe).
                current_base_height = max(current_base_height, vessel.deck_height)
            
            # Calculate VCG for this specific container
            # VCG = Base + Half Height
            half_height = slot.container.height / 2.0
            vcg_container = current_base_height + half_height
            
            w = slot.container.weight
            cargo_weight += w
            cargo_moment += w * vcg_container
            
            # Update the base for the NEXT container in this stack
            current_base_height += slot.container.height

    # --- Hydrostatic Calculation ---
    disp = vessel.lightship_weight + cargo_weight
    
    if disp == 0:
        return 20.0  # Fallback for empty vessel

    # VCG = (Lightship Moment + Cargo Moment) / Total Displacement
    vcg = ((vessel.lightship_weight * vessel.lightship_vcg) + cargo_moment) / disp
    
    # Look up KM (Keel to Metacenter) from hydro tables
    # (Assuming vessel.hydro_disp and vessel.hydro_km are sorted lists/arrays)
    km = np.interp(disp, vessel.hydro_disp, vessel.hydro_km)
    
    return km - vcg


def calculate_cost(vessel: Vessel, leftovers: List[Container] = []) -> float:
    # 1. Safety (GM)
    gm = calculate_gm(vessel)
    cost_gm = (MIN_GM - gm) * W_GM_FAIL if gm < MIN_GM else 0.0

    # 2. Efficiency (Rehandles)
    cost_rehandles = vessel.calculate_rehandles() * W_REHANDLE

    # 3. Balance (Moments)
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
        # FIX 4: Guard against None returns from moment calculations.
        # calculate_*_moment() already returns 0.0 for empty vessels, but
        # this makes the contract explicit and prevents log() from crashing.
        bay_m = self.vessel.calculate_bay_moment() or 0.0
        row_m = self.vessel.calculate_row_moment() or 0.0
        tier_m = self.vessel.calculate_tier_moment() or 0.0
        self.moments = (bay_m, row_m, tier_m)
        self.gm = calculate_gm(self.vessel)

        c_gm = (MIN_GM - self.gm) * W_GM_FAIL if self.gm < MIN_GM else 0.0
        c_re = self.rehandles * W_REHANDLE
        c_bal = (abs(self.moments[0]) + abs(self.moments[1])) * W_BALANCE
        self.total_cost = c_gm + c_re + c_bal

    @classmethod
    def header(cls):
        return "bays,rows,tiers,cost,lightVCG,lightWeight,hydroDisp(x),hydroKM(y),rehandles,bayMoment,rowMoment,tierMoment,gm"

    def log(self):
        return (
            f'{self.vessel.bays},{self.vessel.rows},{self.vessel.tiers},'
            f'{self.total_cost},{self.vessel.lightship_vcg},{self.vessel.lightship_weight},'
            f'"{self.vessel.hydro_km}","{self.vessel.hydro_disp}",'
            f'{self.rehandles},{self.bay_moment},{self.row_moment},{self.tier_moment},{self.gm}'
        )

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
    Parses a vessel definition file (Larsen & Pacino format).
    Robustly handles both inline data and header-only lines.
    """
    bays, rows, tiers = 0, 0, 0
    hydro_disp: List[float] = []
    hydro_km: List[float] = []
    
    total_bay_weight = 0.0
    total_bay_moment = 0.0
    above_deck_vcgs: List[float] = []
    slot_definitions = []

    # Context trackers
    current_bay = -1
    current_row = -1
    current_section_vcg = 0.0
    section_hydro = False

    with open(filepath, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith("//"):
            continue

        # --- A. Global Dimensions ---
        if line.startswith("# Ship:"):
            try:
                parts = lines[i+1].split()
                bays = int(parts[0])
                rows = int(parts[1])
                tiers = int(parts[2])
            except (IndexError, ValueError):
                pass
            section_hydro = False
            continue

        # --- B. Hydrostatics ---
        if line.startswith("## HydroPoints:"):
            section_hydro = True
            continue
        
        # --- C. Bay Context ---
        if line.startswith("## Bay:"):
            section_hydro = False
            parts = line.split()
            # Check if inline: "## Bay: 0 ..." vs Header: "## Bay: index ..."
            if len(parts) > 2 and parts[2].isdigit():
                current_bay = int(parts[2])
                # If data is inline (rare variant)
                if len(parts) >= 7:
                    w = float(parts[5])
                    vcg = float(parts[6])
                    total_bay_weight += w
                    total_bay_moment += (w * vcg)
            else:
                # Data is on NEXT line
                try:
                    data_line = lines[i+1].strip()
                    d_parts = data_line.split()
                    if d_parts and d_parts[0].replace('-','').isdigit():
                        current_bay = int(d_parts[0])
                        # Get weight/vcg from data line
                        if len(d_parts) >= 7:
                            w = float(d_parts[5])
                            vcg = float(d_parts[6])
                            total_bay_weight += w
                            total_bay_moment += (w * vcg)
                except (IndexError, ValueError):
                    pass
            continue

        # --- D. Stack (Row) Context ---
        if line.startswith("### Stack:"):
            parts = line.split()
            # Check inline vs next line
            if len(parts) > 2 and parts[2].isdigit():
                current_row = int(parts[2])
            else:
                try:
                    data_line = lines[i+1].strip()
                    d_parts = data_line.split()
                    if d_parts and d_parts[0].replace('-','').isdigit():
                        current_row = int(d_parts[0])
                except (IndexError, ValueError):
                    pass
            continue

        # --- E. Section VCG (Above/Below Deck) ---
        if line.startswith("#### AboveDeck:") or line.startswith("#### BelowDeck:"):
            try:
                # Data is ALWAYS on next line for these blocks in standard files
                data_line = lines[i+1].strip()
                d_parts = data_line.split()
                # Format: id maxH maxW20 maxW40 vcg
                vcg = float(d_parts[-1])
                current_section_vcg = vcg
                
                if line.startswith("#### AboveDeck:"):
                    above_deck_vcgs.append(vcg)
            except (IndexError, ValueError):
                pass
            continue

        # --- F. Tiers (Slots) ---
        # The header "#### Cell: tier reefer" usually precedes the list
        if line.startswith("#### Cell:"):
            continue

        # Hydro Data Processing
        if section_hydro:
            parts = line.split()
            if len(parts) == 4 and parts[0].replace('.', '', 1).replace('-', '').isdigit():
                hydro_disp.append(float(parts[0]))
                hydro_km.append(float(parts[3]))
            continue

        # Slot Processing (if line starts with a digit and we have context)
        # We must ensure we aren't reading a line we already peeked at (like Bay data)
        # But since headers are distinct, we only process raw numbers here.
        if current_bay >= 0 and current_row >= 0 and line[0].isdigit():
            # Distinguish between "Bay Definition Data" and "Tier Data"
            # Bay/Stack definitions usually have many columns (LCG, Shear, etc.)
            # Tier definitions usually have 2 columns: "tier reefer"
            parts = line.split()
            
            if len(parts) == 2:
                try:
                    tier = int(parts[0])
                    # Store slot
                    slot_definitions.append({
                        'bay': current_bay,
                        'row': current_row,
                        'tier': tier,
                        'vcg': current_section_vcg
                    })
                except ValueError:
                    pass

    # --- Reconstruction ---
    lightship_vcg = total_bay_moment / total_bay_weight if total_bay_weight > 0 else 15.0
    deck_height = min(above_deck_vcgs) if above_deck_vcgs else 15.0

    # Determine max dimensions from observed slots
    if slot_definitions:
        bays = max(bays, max(s['bay'] for s in slot_definitions) + 1)
        rows = max(rows, max(s['row'] for s in slot_definitions) + 1)
        tiers = max(tiers, max(s['tier'] for s in slot_definitions) + 1)

    v = Vessel(
        bays=bays, 
        rows=rows, 
        tiers=tiers,
        lightship_weight=total_bay_weight,
        lightship_vcg=lightship_vcg,
        hydro_disp=hydro_disp,
        hydro_km=hydro_km,
        deck_height=deck_height
    )

    # Populate
    for s_def in slot_definitions:
        coord = SlotCoord(s_def['bay'], s_def['row'], s_def['tier'])
        if coord in v.slots:
            v.slots[coord].vcg = s_def['vcg']

    return v


def parse_benchmark_containers(filepath: Path) -> List[Container]:
    """
    Parses VLHigh1.txt for container weights, heights, and load list.

    Transport type table format:
        id  length=(20|40)  weight  type=(DC|RC|HC|HR)

    Height is derived from the type column:
        HC / HR  →  2.9 m  (High Cube)
        DC / RC  →  2.6 m  (Standard)
    """
    CONTAINER_HEIGHT_STANDARD = 2.6
    CONTAINER_HEIGHT_HC = 2.9
    _HIGH_CUBE_TYPES = {"HC", "HR"}

    containers: List[Container] = []
    transport_props: Dict[int, tuple[float, float]] = {}

    with open(filepath, "r") as f:
        lines = f.readlines()

    mode = None

    for line in lines:
        line = line.strip()
        if not line or line.startswith("//"):
            continue

        if "# Transport type" in line:
            mode = "WEIGHTS"
            continue
        elif "# Container" in line:
            mode = "LOADLIST"
            continue

        parts = line.split()
        if not parts[0].isdigit():
            continue

        try:
            if mode == "WEIGHTS":
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
