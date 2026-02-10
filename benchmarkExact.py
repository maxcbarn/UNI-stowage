from common import *
from exact import *
import timeit
import os
import re

ship = []

def main():
    with open("logs/exact.csv", "w") as log:
        log.write("executeTime(s), containersQuantity, bays, rows, tiers, cost, rehandles, bayMoment, rowMoment, tierMoment\n")
        
        containerToTest = map(lambda x: os.path.join("containers", x) , os.listdir("containers") )
        containers = []
        for container in containerToTest:
            if int( re.match(r"^.*containers-(\d+)\.txt", container).group(1) ) > 30:
                continue
            
            with open(container, "r") as file:
                containers = eval(file.read())
            
            def run(  ):
                global ship
                ship = solve_stowage_3d_min_rehandles_with_cog( containers , 5 , 2 , 6 )
            timeSeconds = timeit.timeit( run, number=1)
            cost = calculate_cost( * ship_to_vessel( ship ) )
            report = CostReport( ship )
            log.write(f"{timeSeconds},{len(containers)},{len(ship)},{len(ship[0])},{len(ship[0][0])},{cost},{report.rehandles},{report.bay_moment},{report.row_moment},{report.tier_moment}\n")
    
    
main()