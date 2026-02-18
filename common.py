from __future__ import annotations
import json

import numpy as np
from typing import List, Dict, Optional, Tuple, TypeVar, Generic, TypedDict
import random
from itertools import product
from dataclasses import dataclass, field
from pathlib import Path



LIGHTSHIP_WEIGHT = 60787.0


LIGHTSHIP_VCG = 18.0


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


AVG_CONTAINER_WEIGHT = 14.0  
CONTAINER_HEIGHT = 2.6       
MIN_GM = 1.0                 







DECK_HEIGHT = 13.0  


W_REHANDLE = 10000.0    
W_GM_FAIL = 50000.0     
W_BALANCE = 1.0         
W_LEFTOVER = 100000.0   

W_STACK_BONUS = 5_000.0     
W_CRANE_BALANCE = 500.0     


W_WEIGHT_INVERSION = 5_000.0  





W_MOMENT_TIER = 200.0   
W_MOMENT_ROW  = 300.0   

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






@dataclass
class Vessel:
    bays: int = 0
    rows: int = 0
    tiers: int = 0
    lightship_weight: float = LIGHTSHIP_WEIGHT
    lightship_vcg: float = LIGHTSHIP_VCG
    hydro_disp: List[float] = field(default_factory=lambda: HYDRO_X)
    hydro_km: List[float] = field(default_factory=lambda: HYDRO_Y)
    
    
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
            
            "contCount": self.containerAmount,
            "bays": self.bays,
            "rows": self.rows,
            "tiers": self.tiers,
            
            
            "lightship_weight": self.lightship_weight,
            "lightship_vcg": self.lightship_vcg,
            "deck_height": self.deck_height,
            
            
            "hydro_disp": self.hydro_disp,
            "hydro_km": self.hydro_km,
            
            
            "cost": rep.total_cost,
            "rehandles": rep.rehandles,
            "gm": rep.gm,
            "rowMoment": rep.row_moment,
            "bayMoment": rep.bay_moment,
            "tierMoment": rep.tier_moment,
            
            
            "slots": [
                {
                    "bay": coord.bay,
                    "row": coord.row,
                    "tier": coord.tier,
                    "vcg": slot.vcg, 
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
    
    
    TIER_ON_DECK_START = 80   
    TANK_TOP_HEIGHT = 0.5     
    
    cargo_weight = 0.0
    cargo_moment = 0.0

    
    stacks: Dict[Tuple[int, int], List[Slot]] = {}
    for slot in vessel.slots.values():
        if slot.container:
            key = (slot.bay, slot.row)
            stacks.setdefault(key, []).append(slot)

    for stack_slots in stacks.values():
        
        stack_slots.sort(key=lambda s: s.tier)
        
        
        
        current_base_height = TANK_TOP_HEIGHT
        
        for slot in stack_slots:
            
            if slot.tier >= TIER_ON_DECK_START:
                
                
                
                current_base_height = max(current_base_height, vessel.deck_height)
            
            
            
            half_height = slot.container.height / 2.0
            vcg_container = current_base_height + half_height
            
            w = slot.container.weight
            cargo_weight += w
            cargo_moment += w * vcg_container
            
            
            current_base_height += slot.container.height

    
    disp = vessel.lightship_weight + cargo_weight
    
    if disp == 0:
        return 20.0  

    
    vcg = ((vessel.lightship_weight * vessel.lightship_vcg) + cargo_moment) / disp
    
    
    
    km = np.interp(disp, vessel.hydro_disp, vessel.hydro_km)
    
    return km - vcg


def calculate_cost(vessel: Vessel, leftovers: List[Container] = []) -> float:
    
    gm = calculate_gm(vessel)
    cost_gm = (MIN_GM - gm) * W_GM_FAIL if gm < MIN_GM else 0.0

    
    cost_rehandles = vessel.calculate_rehandles() * W_REHANDLE

    
    cost_balance = (abs(vessel.calculate_bay_moment()) +
                    abs(vessel.calculate_row_moment())) * W_BALANCE

    
    cost_leftover = len(leftovers) * W_LEFTOVER

    return cost_gm + cost_rehandles + cost_balance + cost_leftover




@dataclass
class CostReport:
    vessel: Vessel
    rehandles: int = field(init=False)
    moments: Tuple[float, float, float] = field(init=False)
    gm: float = field(init=False)
    total_cost: float = field(init=False)

    def __post_init__(self):
        self.rehandles = self.vessel.calculate_rehandles()
        
        
        
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

        
        if line.startswith("
            try:
                parts = lines[i+1].split()
                bays = int(parts[0])
                rows = int(parts[1])
                tiers = int(parts[2])
            except (IndexError, ValueError):
                pass
            section_hydro = False
            continue

        
        if line.startswith("
            section_hydro = True
            continue
        
        
        if line.startswith("
            section_hydro = False
            parts = line.split()
            
            if len(parts) > 2 and parts[2].isdigit():
                current_bay = int(parts[2])
                
                if len(parts) >= 7:
                    w = float(parts[5])
                    vcg = float(parts[6])
                    total_bay_weight += w
                    total_bay_moment += (w * vcg)
            else:
                
                try:
                    data_line = lines[i+1].strip()
                    d_parts = data_line.split()
                    if d_parts and d_parts[0].replace('-','').isdigit():
                        current_bay = int(d_parts[0])
                        
                        if len(d_parts) >= 7:
                            w = float(d_parts[5])
                            vcg = float(d_parts[6])
                            total_bay_weight += w
                            total_bay_moment += (w * vcg)
                except (IndexError, ValueError):
                    pass
            continue

        
        if line.startswith("
            parts = line.split()
            
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

        
        if line.startswith("
            try:
                
                data_line = lines[i+1].strip()
                d_parts = data_line.split()
                
                vcg = float(d_parts[-1])
                current_section_vcg = vcg
                
                if line.startswith("
                    above_deck_vcgs.append(vcg)
            except (IndexError, ValueError):
                pass
            continue

        
        
        if line.startswith("
            continue

        
        if section_hydro:
            parts = line.split()
            if len(parts) == 4 and parts[0].replace('.', '', 1).replace('-', '').isdigit():
                hydro_disp.append(float(parts[0]))
                hydro_km.append(float(parts[3]))
            continue

        
        
        
        if current_bay >= 0 and current_row >= 0 and line[0].isdigit():
            
            
            
            parts = line.split()
            
            if len(parts) == 2:
                try:
                    tier = int(parts[0])
                    
                    slot_definitions.append({
                        'bay': current_bay,
                        'row': current_row,
                        'tier': tier,
                        'vcg': current_section_vcg
                    })
                except ValueError:
                    pass

    
    lightship_vcg = total_bay_moment / total_bay_weight if total_bay_weight > 0 else 15.0
    deck_height = min(above_deck_vcgs) if above_deck_vcgs else 15.0

    
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

        if "
            mode = "WEIGHTS"
            continue
        elif "
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
