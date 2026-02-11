from common import *
from probabilistic import *
from genetic import *
from aproximative import *
import timeit
import os
import re

ship = []
vessel = None

def Genetic( containers, num_bays, num_rows, max_tiers ):
    global ship
    def run():
        global ship
        ship = solve_stowage_genetic( containers, num_bays, num_rows, max_tiers )
    timeSeconds = timeit.timeit( run, number=1)
    
    with open("logs/genetic.csv", "a") as log:
        cost = calculate_cost( * ship_to_vessel( ship ) )
        report = CostReport( ship )
        log.write(f"{timeSeconds},{len(containers)},{num_bays},{num_rows},{max_tiers},{cost},{report.rehandles},{report.bay_moment},{report.row_moment},{report.tier_moment}\n")
    print( f"Finished Genetic for {len(containers)} containers in {timeSeconds:.2f}s" )
    return 

def Probabilistic( containers, num_bays, num_rows, max_tiers ):
    global vessel
    def run():
        global vessel
        def ri(vals: List[int], f: bool = False):
            if len(vals) == 1:
                return Range(vals[0], vals[0] if f else vals[0]+1)
            return Range(vals[0], vals[1])
        inputVessel = Vessel( num_bays , num_rows, max_tiers )
        vessel , _ = mcts_search( inputVessel , cont_to_containter( containers ) )
    timeSeconds = timeit.timeit( run, number=1)
    
    with open("logs/probabilistic.csv", "a") as log:
        cost = calculate_cost( vessel , [] )
        report = CostReport( vessel_to_ship( vessel ) )
        log.write(f"{timeSeconds},{len(containers)},{num_bays},{num_rows},{max_tiers},{cost},{report.rehandles},{report.bay_moment},{report.row_moment},{report.tier_moment}\n")
    print( f"Finished Probabilistic for {len(containers)} containers in {timeSeconds:.2f}s" )
    return 

def Aproximative( containers, num_bays, num_rows, max_tiers ):
    global vessel
    def run():
        global vessel
        def ri(vals: List[int], f: bool = False):
            if len(vals) == 1:
                return Range(vals[0], vals[0] if f else vals[0]+1)
            return Range(vals[0], vals[1])
        inputVessel = Vessel( num_bays , num_rows, max_tiers )
        vessel , _ = heuristic_solver( cont_to_containter( containers ) , inputVessel )
    timeSeconds = timeit.timeit( run, number=1)
    
    with open("logs/aproximative.csv", "a") as log:
        cost = calculate_cost( vessel , [] )
        report = CostReport( vessel_to_ship( vessel ) )
        log.write(f"{timeSeconds},{len(containers)},{num_bays},{num_rows},{max_tiers},{cost},{report.rehandles},{report.bay_moment},{report.row_moment},{report.tier_moment}\n")
    print( f"Finished Aproximative for {len(containers)} containers in {timeSeconds:.2f}s" ) 
    return 

def main():
    executions = 0;
    bays = 0
    rows = 0
    tier = 0
    with open("logs/probabilistic.csv", "w") as log:
        log.write("executeTime(s), containersQuantity, bays, rows, tiers, cost, rehandles, bayMoment, rowMoment, tierMoment\n")
    with open("logs/genetic.csv", "w") as log:
        log.write("executeTime(s), containersQuantity, bays, rows, tiers, cost, rehandles, bayMoment, rowMoment, tierMoment\n")
    with open("logs/aproximative.csv", "w") as log:
        log.write("executeTime(s), containersQuantity, bays, rows, tiers, cost, rehandles, bayMoment, rowMoment, tierMoment\n")
    containerToTest = map(lambda x: os.path.join("containers", x) , os.listdir("containers") )
    containers = []
    for container in containerToTest:
        if int( re.match(r"^.*containers-(\d+)\.txt", container).group(1) ) <= 30:
            executions = 32
            bays = 5
            rows = 2
            tier = 6
        if int( re.match(r"^.*containers-(\d+)\.txt", container).group(1) ) > 140:
            executions = 16
            bays = 10
            rows = 5
            tier = 5
        if int( re.match(r"^.*containers-(\d+)\.txt", container).group(1) ) > 250:
            executions = 8
            bays = 10
            rows = 10
            tier = 10
        
        with open(container, "r") as file:
            containers = eval(file.read())
        
        for x in range(0, executions):
            Genetic( containers , bays , rows , tier )
            Probabilistic( containers , bays , rows , tier )
            Aproximative( containers , bays , rows , tier ) 
            
            
    
main()