from common import *
from exact import *
from aproximative import *
from genetic import *


def main():
     # Example containers (id, weight, dest)
    containers = [
        {'id': 1,  'weight': 4.0, 'dest': 1},
        {'id': 2,  'weight': 6.0, 'dest': 1},
        {'id': 3,  'weight': 20.0, 'dest': 1},
        {'id': 4,  'weight': 5.0, 'dest': 1},
        {'id': 5,  'weight': 3.0, 'dest': 1},
        {'id': 6,  'weight': 8.0, 'dest': 3},
        {'id': 7,  'weight': 80.0, 'dest': 2},
        {'id': 8,  'weight': 16.0, 'dest': 1},
        {'id': 9,  'weight': 8.0, 'dest': 1},
        {'id': 10, 'weight': 8.0, 'dest': 1},
    ]

    num_bays = 5
    num_rows = 2
    max_tiers = 8

    containerR = ContainerRamdom(10);
    print( containerR )
    # You can change ALPHA/BETA to tune priority between rehandles and balance
    res = solve_stowage_3d_min_rehandles_with_cog(
        containerR, num_bays, num_rows, max_tiers,
        ALPHA=1.0, BETA=0.05
    ) 
    print( res );
    res = solve_stowage_genetic(containerR, num_bays, num_rows, max_tiers)
    print( res );
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
main()