from typing import List, Dict, Tuple, Optional
import random

def BuildStacks(num_bays: int, num_rows: int):
    stacks = []
    for bay in range(num_bays):
        for row in range(num_rows):
            stacks.append((bay, row))
    return stacks
    
def RehandlesNumber( ship: List[List[List[dict[int,float,int]]]] ):
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

def BayMoment(ship: List[List[List[dict[int,float,int]]]] , baySize: int ):
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


def ContainerRamdom(number: int) -> dict[int,float,int]:

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
