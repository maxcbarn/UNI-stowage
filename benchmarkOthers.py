from collections import defaultdict
import multiprocessing
from typing import Callable, Dict, List, Set, Tuple, TypedDict
from common import CostReport, Vessel, Container, conts_to_containers, parse_benchmark_containers, parse_benchmark_vessel
from probabilistic import mcts_search
from genetic import solve_stowage_genetic
from aproximative import heuristic_solver
from exact import solve_stowage_lp
import timeit
import os
import re
import sys
import time
import pandas as pd

from pathlib import Path

ship = []
vessel = None


def Genetic(containers: List[Container], vessel: Vessel):
    def run():
        nonlocal vessel, containers
        vessel = solve_stowage_genetic(containers, vessel)
    timeSeconds = timeit.timeit(run, number=1)

    cost = log("genetic", vessel, [], containers, timeSeconds)
    print(
        f"Finished Genetic for {vessel.containerAmount}/{len(containers)} containers in {timeSeconds:.2f}s")

    return cost


def Probabilistic(containers: List[Container], vessel: Vessel):
    def run():
        nonlocal vessel, containers
        vessel_temp, _ = mcts_search(vessel, containers, 100)
        if vessel_temp:
            vessel = vessel_temp

    timeSeconds = timeit.timeit(run, number=1)

    cost = log("probabilistic", vessel, [], containers, timeSeconds)

    print(
        f"Finished Probabilistic for {vessel.containerAmount}/{len(containers)}containers in {timeSeconds:.2f}s")

    return cost


def Aproximative(containers: List[Container], vessel: Vessel):
    missing: List[Container] = []

    def run():
        nonlocal vessel, containers, missing

        vessel, missing = heuristic_solver(containers, vessel)
    timeSeconds = timeit.timeit(run, number=1)

    cost = log("aproximative", vessel, missing, containers, timeSeconds)

    print(
        f"Finished Aproximative for {vessel.containerAmount}:{len(missing)}/{len(containers)} containers in {timeSeconds:.2f}s")

    return cost


def Exact(containers: List[Container], vessel: Vessel):
    def run():
        nonlocal vessel, containers

        vessel, _ = solve_stowage_lp(containers, vessel)
    timeSeconds = timeit.timeit(run, number=1)

    cost = log("aproximative", vessel, [], containers, timeSeconds)

    print(
        f"Finished Aproximative for {vessel.containerAmount}/{len(containers)} containers in {timeSeconds:.2f}s")

    return cost


def log(name: str, vessel: Vessel, missing: List[Container], containers: List[Container], timeSeconds: float) -> float:
    with open(f"logs/{name}.csv", "a") as log:
        report = CostReport(vessel)
        log.write(f"{timeSeconds},{len(containers)},{report.log()}\n")
    with open(f"logs/{name}.json", "a") as json:
        json.write(',' + vessel.JSON_str() + '\n')
    return report.total_cost


type SolutionFunc = Callable[[List[Container], Vessel], float]


class ResultEntry(TypedDict):
    algo: str
    time: float
    cost: float
    config: str


def main():
    solutions: Dict[str, SolutionFunc] = {
        "Genetic": Genetic,
        "Aproximative": Aproximative,
        "Probabilistic": Probabilistic,
        "Exact": Exact,
    }

    sols = sys.argv[1:]
    if sols[0].lower() == 'all':
        sols = solutions.keys()
    elif sols[0].lower().startswith('others'):
        sols = solutions.keys() - ['Exact']
    else:
        for i in range(len(sols)):
            sols[i] = sols[i].capitalize()
            for key in solutions.keys():
                if key.startswith(sols[i]):
                    sols[i] = key
                    break

    executions = 0
    bays = 0
    rows = 0
    tier = 0
    os.makedirs("logs", exist_ok=True)

    skip: Dict[str, Dict[Tuple[int, int, int, int], int]
               ] = defaultdict(lambda: defaultdict(lambda: 0))
    for sol in sols:
        log_path = Path(f"logs/{sol.lower()}.csv")

        if (log_path.exists()):
            def getTuple(x):
                return x.containersQuantity, x.bays, x.rows, x.tiers

            for entry in map(lambda x: getTuple(
                    x), pd.read_csv(log_path).itertuples(index=True, name='Pandas')):
                skip[sol][entry] += 1
            print(skip)

        else:
            with open(f"logs/{sol.lower()}.csv", "w") as log:
                log.write(
                    f"executeTime(s),containersQuantity,{CostReport.header()}\n")

        json_path = Path(f"logs/{sol.lower()}.json")
        init = "[{}\n"
        if not json_path.exists():
            with open(json_path, 'w') as json:
                json.write(init)

    containerToTest = list(map(lambda x: os.path.join(
        "containers", x), os.listdir("containers")))
    containers = []
    for container in containerToTest:
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

        executions = 1

        vessel = Vessel(bays=bays, rows=rows, tiers=tier)

        with open(container, "r") as file:
            containers = conts_to_containers(eval(file.read()))

        for arg in sols:
            if arg == "Exact" and len(containers) > 15:
                sols.remove(arg)
            executions -= skip[arg][(len(containers),
                                     vessel.bays, vessel.rows, vessel.tiers)]
            for _ in range(executions):
                solutions[arg](containers, vessel)

    # Paths to your benchmark folders
    vessel_folder = "Stowage-Planning-Benckmark/vessel_data"
    instance_folder = "Stowage-Planning-Benckmark/container_instances"

    # Track results for sorting
    results: List[ResultEntry] = []

    # Get all combinations
    v_files = [Path(r) / f for r, _, fs in os.walk(vessel_folder)
               for f in fs if f.endswith(".txt")]

    i_files = [Path(r) / f for r, _, fs in os.walk(instance_folder)
               for f in fs if f.endswith(".txt")]

    inputs: List[Tuple[List[Container], Vessel, str, str]] = []
    for v_path in v_files:
        vessel = parse_benchmark_vessel(v_path)
        for i_path in i_files:
            conts = parse_benchmark_containers(i_path)

            inputs.append((conts, vessel, v_path.parts[-1], i_path.parts[-1]))

    sorted_inputs = sorted(
        inputs, key=lambda inp: len(inp[0]) * inp[1].capacity)

    for inp in sorted_inputs:
        executions = 1
        conts, vessel, v, i = inp
        if vessel.capacity < len(conts):
            print(
                f"Skippng: {v} + {i} -> {len(conts)} < {vessel.bays} * {vessel.rows} * {vessel.tiers} ({vessel.capacity})")
            continue
        for sol in sols:
            print(
                f"Running {sol}: {v} + {i} -> ({len(conts)}, {vessel.bays} * {vessel.rows} * {vessel.tiers})")

            elapsed = 0
            cost = 0
            executions -= skip[sol][(len(conts),
                                     vessel.bays, vessel.rows, vessel.tiers)]

            for _ in range(executions):
                start = time.perf_counter()
                cost += solutions[sol](conts, vessel)
                elapsed += time.perf_counter() - start

            elapsed /= executions
            cost /= executions

            results.append({
                "algo": sol,
                "time": elapsed,
                "cost": cost,
                "config": f"{v} | {i}"
            })

    # Sort results by execution time
    sorted_res = sorted(results, key=lambda x: x['time'])

    print("\n" + "="*70)
    print(f"{'Solution':<15} | {'Execution (s)':<15} | {'Cost':<10} | {'Configuration'}")
    print("-" * 70)
    for res in sorted_res:
        print(
            f"{res['algo']:<15} | {res['time']:<15.4f} | {res['cost']:<10.4f} | {res['config']}")

    for sol in sols:
        with open(f"logs/{sol.lower()}.json", "a") as log:
            log.write("]\n")


if __name__ == "__main__":
    multiprocessing.freeze_support()

    main()
