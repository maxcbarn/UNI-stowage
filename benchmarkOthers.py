from collections import defaultdict
from typing import Callable
from common import *
from probabilistic import *
from genetic import *
from temp import DEAPSolver
from aproximative import *
import timeit
import os
import re
import sys
import time

from pathlib import Path

ship = []
vessel = None


def Genetic(containers, num_bays, num_rows, max_tiers):
    def run():
        global ship
        ship = solve_stowage_genetic(containers, num_bays, num_rows, max_tiers)
    timeSeconds = timeit.timeit(run, number=1)

    with open("logs/genetic.csv", "a") as log:
        cost = calculate_cost(* ship_to_vessel(ship))
        report = CostReport(ship)
        log.write(f"{timeSeconds},{len(containers)},{num_bays},{num_rows},{max_tiers},{cost},{report.rehandles},{report.bay_moment},{report.row_moment},{report.tier_moment}\n")
    print(
        f"Finished Genetic for {count_cont_ship(ship)}/{len(containers)} containers in {timeSeconds:.2f}s")

    return cost


def Temp(conts: List[Cont], num_bays: int, num_rows: int, max_tiers: int):
    # We use a nonlocal variable to capture the result from the inner run() function
    best_vessel = None
    containers = cont_to_containter(conts)

    def run():
        nonlocal best_vessel
        nonlocal containers

        # 1. Create the empty vessel template based on dimensions
        template = Vessel(num_bays, num_rows, max_tiers)

        # 2. Instantiate the Solver
        # Adjust pop_size and generations to balance Speed vs. Quality
        # larger pop_size = better results but slower
        solver = DEAPSolver(
            vessel_template=template,
            containers=containers,
            pop_size=50,
        )

        # 3. Run the Genetic Algorithm
        # We discard the returned cost here, as we recalculate it later for the log
        best_vessel, _ = solver.run(generations=50, use_multiprocessing=True)

    # Time the execution
    timeSeconds = timeit.timeit(run, number=1)

    # --- Post-Processing for Logging ---

    # 1. Convert the resulting Vessel object back to a "Ship" (3D List)
    # This is required because CostReport expects a 3D list.
    ship = vessel_to_ship(best_vessel)

    # 2. Recalculate cost for logging
    # Note: ship_to_vessel returns (Vessel, leftovers=[]).
    # If accurate leftover tracking is critical, calculate leftovers by comparing
    # 'containers' vs 'ship' contents manually.
    vessel_for_calc, _ = ship_to_vessel(ship)

    # We pass the full original list to check which ones are missing (leftovers)
    # This ensures the cost function penalizes any containers the GA failed to load.
    stowed_ids = set()
    for slot in vessel_for_calc.slots.values():
        if slot.container:
            stowed_ids.add(slot.container.id)

    real_leftovers = [c for c in containers if c.id not in stowed_ids]
    cost = calculate_cost(vessel_for_calc, real_leftovers)

    # 3. Write to Log
    with open("logs/temp.csv", "a") as log:
        report = CostReport(ship)
        log.write(f"{timeSeconds},{len(containers)},{num_bays},{num_rows},{max_tiers},{cost},{report.rehandles},{report.bay_moment},{report.row_moment},{report.tier_moment}\n")

    print(
        f"Finished Genetic for {len(containers)} containers in {timeSeconds:.2f}s")

    return cost


def Probabilistic(containers, num_bays, num_rows, max_tiers):
    def run():
        global vessel
        inputVessel = Vessel(num_bays, num_rows, max_tiers)
        vessel, _ = mcts_search(
            inputVessel, cont_to_containter(containers), 100)
    timeSeconds = timeit.timeit(run, number=1)

    with open("logs/probabilistic.csv", "a") as log:
        cost = calculate_cost(vessel, [])
        report = CostReport(vessel_to_ship(vessel))
        log.write(f"{timeSeconds},{len(containers)},{num_bays},{num_rows},{max_tiers},{cost},{report.rehandles},{report.bay_moment},{report.row_moment},{report.tier_moment}\n")
    print(
        f"Finished Probabilistic for {vessel.containerAmount}/{len(containers)}containers in {timeSeconds:.2f}s")

    return cost


def Aproximative(containers, num_bays, num_rows, max_tiers):
    def run():
        global vessel

        inputVessel = Vessel(num_bays, num_rows, max_tiers)

        cs = cont_to_containter(containers)
        vessel, missing = heuristic_solver(cs, inputVessel)
    timeSeconds = timeit.timeit(run, number=1)

    with open("logs/aproximative.csv", "a") as log:
        cost = calculate_cost(vessel, [])
        report = CostReport(vessel_to_ship(vessel))
        log.write(f"{timeSeconds},{len(containers)},{num_bays},{num_rows},{max_tiers},{cost},{report.rehandles},{report.bay_moment},{report.row_moment},{report.tier_moment}\n")
    print(
        f"Finished Aproximative for {vessel.containerAmount}/{len(containers)} containers in {timeSeconds:.2f}s")

    return cost


def parse_benchmark_vessel(filepath: Path) -> Tuple[int, int, int]:
    """
    State-machine parser to handle hierarchical benchmark files.
    Extracts global dimensions (bays, stacks, tiers) from the vessel layout.

    File format (vessel_L.txt):
        # Ship: bays stacks tiers tcgTollerance
        24 22 21 0.100
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("# Ship:"):
            # Data is on the NEXT line; split() handles any whitespace/CRLF
            data = lines[i + 1].split()
            # Columns: bays  stacks  tiers  tcgTolerance
            #   idx:    0      1       2        3
            return int(data[0]), int(data[1]), int(data[2])

    # Fallback – header not found
    return 0, 0, 0


# ---------------------------------------------------------------------------
# VLHigh1.txt  ──  parse transport-type weights + container load list
# ---------------------------------------------------------------------------
def parse_benchmark_containers(filepath: Path) -> List[Cont]:
    """
    Parses VLHigh1.txt.

    File sections
    -------------
    1. # Parameters: nPorts nContainers          ← ignored
       14 7248

    2. # Transport type: id length=(20,40) weight type=(DC,RC,HC,HR)
       0  20  3  DC                              ← id  length  weight  type
       ...

    3. # Container: startPort endPort typeId [bay stack tier slot]
       0  3  3  1  5  12  1                      ← 7-column rows (with position)
       0  3  3                                   ← 3-column rows (no position)

    Bugs fixed vs. original
    -----------------------
    * "# Containers" → "# Container"   (the header has no trailing 's')
    * CRLF line endings are handled implicitly via strip()
    * parts[0].isdigit() guard kept; it correctly skips the comment header line
    """
    containers: List[Cont] = []
    transport_weights: dict = {}

    with open(filepath, 'r') as f:
        lines = f.readlines()

    mode = None

    for line in lines:
        line = line.strip()
        if not line or line.startswith("//"):
            continue

        # ── Section switches ──────────────────────────────────────────────
        if "# Transport type" in line:
            mode = "WEIGHTS"
            continue
        elif "# Container" in line:          # FIX: was "# Containers" — no 's'
            mode = "LOADLIST"
            continue

        parts = line.split()
        if not parts:
            continue

        try:
            if mode == "WEIGHTS":
                # Format: id  length  weight  type
                #  idx:   0    1       2       3
                t_id = int(parts[0])
                weight = float(parts[2])
                transport_weights[t_id] = weight

            elif mode == "LOADLIST":
                # Skip comment / header lines (e.g. "# Parameters: 14 7248")
                if not parts[0].isdigit():
                    continue

                # Columns present in BOTH row formats:
                #   startPort(0)  endPort(1)  typeId(2)
                # Optional extra columns (7-col rows only):
                #   bay(3)  stack(4)  tier(5)  slot(6)
                dest_port = int(parts[1])
                type_id = int(parts[2])

                containers.append({
                    "id":     len(containers),
                    "weight": transport_weights[type_id],
                    "dest":   dest_port,
                })

        except (ValueError, IndexError, KeyError):
            # Safely skip any unexpected metadata lines
            continue

    return containers


type SolutionFunc = Callable[[List[Cont], int, int, int], float]


def main():
    solutions: Dict[str, SolutionFunc] = {
        "Genetic": Genetic,
        "Aproximative": Aproximative,
        "Probabilistic": Probabilistic,
        "Temp": Temp,
    }

    sols = sys.argv[1:]
    if sols[0].lower() != 'all':
        for i in range(len(sols)):
            sols[i] = sols[i].capitalize()
            for key in solutions.keys():
                if key.startswith(sols[i]):
                    sols[i] = key
                    break
    else:
        sols = solutions.keys()

    executions = 0
    bays = 0
    rows = 0
    tier = 0
    os.makedirs("logs", exist_ok=True)

    for sol in sols:
        with open(f"logs/{sol.lower()}.csv", "w") as log:
            log.write(
                "executeTime(s), containersQuantity, bays, rows, tiers, cost, rehandles, bayMoment, rowMoment, tierMoment\n")

    containerToTest = list(map(lambda x: os.path.join(
        "containers", x), os.listdir("containers")))
    containers = []
    for container in containerToTest[:2]:
        matchRegex = re.match(r"^.*containers-(\d+)\.txt", container)
        containerAmount = int(matchRegex.group(1)) if matchRegex else 0
        if containerAmount <= 30:
            executions = 32
            bays = 5
            rows = 2
            tier = 6
        if containerAmount >= 50:
            executions = 16
            bays = 10
            rows = 5
            tier = 5
        if containerAmount > 250:
            executions = 8
            bays = 10
            rows = 10
            tier = 10

        with open(container, "r") as file:
            containers = eval(file.read())

        for _ in range(0, 1):
            for arg in sols:
                solutions[arg](containers, bays, rows, tier)

    # Paths to your benchmark folders
    vessel_folder = "Stowage-Planning-Benckmark/vessel_data"
    instance_folder = "Stowage-Planning-Benckmark/container_instances"

    # Track results for sorting
    results = []

    # Get all combinations
    v_files = [Path(r) / f for r, _, fs in os.walk(vessel_folder)
               for f in fs if f.endswith(".txt")]

    i_files = [Path(r) / f for r, _, fs in os.walk(instance_folder)
               for f in fs if f.endswith(".txt")]

    inputs: List[Tuple[List[Cont], int, int, int, int, str, str]] = []
    for v_path in v_files:
        b, r, t = parse_benchmark_vessel(v_path)
        for i_path in i_files:
            conts = parse_benchmark_containers(i_path)

            inputs.append((conts, b, r, t, len(conts) * b * r *
                          t, v_path.parts[-1], i_path.parts[-1]))

    sorted_inputs = sorted(inputs, key=lambda inp: inp[4])

    for inp in sorted_inputs:
        vesselCap = inp[1] * inp[2] * inp[3]
        if vesselCap < len(inp[0]):
            print(
                f"Skippng: {inp[-2]} + {inp[-1]} -> {len(inp[0])} < {inp[1]} * {inp[2]} * {inp[3]} ({vesselCap})")
            continue
        for sol in sols:
            print(
                f"Running {sol}: {inp[-2]} + {inp[-1]} -> ({len(inp[0])}, {inp[1]}, {inp[2]}, {inp[3]}) == {inp[4]}")

            start = time.perf_counter()
            cost = solutions[sol](*inp[:4])
            elapsed = time.perf_counter() - start

            results.append({
                "algo": sol,
                "time": elapsed,
                "cost": cost,
                "config": f"{v_path.name} | {i_path.name}"
            })

    # Sort results by execution time
    sorted_res = sorted(results, key=lambda x: x['time'])

    print("\n" + "="*70)
    print(f"{'Solution':<15} | {'Execution (s)':<15} | {'Cost':<10} | {'Configuration'}")
    print("-" * 70)
    for res in sorted_res:
        print(
            f"{res['algo']:<15} | {res['time']:<15.4f} | {res['cost']:<10.4f} | {res['config']}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
