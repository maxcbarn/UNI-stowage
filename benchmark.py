from collections import defaultdict
import multiprocessing
import random
from typing import Callable, Dict, List, Set, Tuple, TypedDict

from pandas._libs.tslibs.fields import build_field_sarray
from pandas.core.dtypes.cast import find_result_type
from common import CostReport, Vessel, Container, conts_to_containers, parse_benchmark_containers, parse_benchmark_vessel
from probabilistic import solve_rolling_horizon
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


def Genetic(containers: List[Container], vessel: Vessel, exe: int):
    def run():
        nonlocal vessel, containers
        vessel = solve_stowage_genetic(containers, vessel)
    timeSeconds = timeit.timeit(run, number=1)

    cost = log("genetic", vessel, [], containers, timeSeconds, exe)

    return cost


def Probabilistic(containers: List[Container], vessel: Vessel, exe: int):
    def run():
        nonlocal vessel, containers
        vessel_temp, _ = solve_rolling_horizon(vessel, containers)
        if vessel_temp:
            vessel = vessel_temp

    timeSeconds = timeit.timeit(run, number=1)

    cost = log("probabilistic", vessel, [], containers, timeSeconds, exe)

    return cost


def Aproximative(containers: List[Container], vessel: Vessel, exe: int):
    missing: List[Container] = []
    vessel.clear()
    def run():
        nonlocal vessel, containers, missing
        vessel, missing = heuristic_solver(containers, vessel)
    timeSeconds = timeit.timeit(run, number=1)

    cost = log("aproximative", vessel, missing, containers, timeSeconds, exe)

    return cost


def Exact(containers: List[Container], vessel: Vessel, exe: int):
    def run():
        nonlocal vessel, containers

        vessel, _ = solve_stowage_lp(containers, vessel)
    timeSeconds = timeit.timeit(run, number=1)

    cost = log("exact", vessel, [], containers, timeSeconds, exe)


    return cost


EXECUTIONS = 32
BUDGET_SECONDS = 530


cur_exes = EXECUTIONS

def log(name: str, vessel: Vessel, missing: List[Container], containers: List[Container], timeSeconds: float, exe: int) -> float:
    with open(f"logs/{name}.csv", "a") as log:
        report = CostReport(vessel)
        log.write(f"{timeSeconds},{len(containers)},{report.log()}\n")

    if exe == 0:
        with open(f"logs/{name}.json", "a") as json:
            json.write(',' + vessel.JSON_str() + '\n')

    global cur_exes
    print(
        f"\t[{exe+1}/{cur_exes}]Finished ${name.capitalize()} ({report.total_cost}) for {vessel.containerAmount}/{len(containers)} containers in {timeSeconds:.2f}s")

    return report.total_cost


type SolutionFunc = Callable[[List[Container], Vessel, int], float]


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


    prune = True
    sols = sys.argv[1:]
    if sols[0].lower() == 'every':
        sols = list(solutions.keys())
        prune = False
    elif sols[0].lower() == 'all':
        sols = list(solutions.keys())
    elif sols[0].lower().startswith('others'):
        sols = list(solutions.keys())
        sols.remove('Exact')
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

    global cur_exes

    def extractKey(container: str):
        matchRegex = re.match(r"^.*containers-(\d+)\.txt", container)
        if matchRegex:
                 return container, int(matchRegex.group(1))
        return container, 0
    containerToTest = list(map(lambda x: extractKey(os.path.join(
        "containers", x)), os.listdir("containers")))
    containerToTest = sorted(containerToTest, key=lambda x: x[1])
    containers = []
    for container, containerAmount in containerToTest:
        if containerAmount <= 30:
        
            bays = 5
            rows = 2
            tier = 6
        if containerAmount >= 50:
        
            bays = 10
            rows = 5
            tier = 5
        if containerAmount > 250:
        
            bays = 10
            rows = 10
            tier = 10

        executions = EXECUTIONS

        vessel = Vessel(bays=bays, rows=rows, tiers=tier)

        with open(container, "r") as file:
            containers = conts_to_containers(eval(file.read()))

        for arg in sols:
            if arg == "Exact" and len(containers) > 100 and prune:
                sols.remove(arg)
            executions -= skip[arg][(len(containers),
                                     vessel.bays, vessel.rows, vessel.tiers)]
            cur_exes = executions
            for exe in range(executions):
                solutions[arg](containers, vessel, exe)

    
    vessel_folder = "Stowage-Planning-Benckmark/vessel_data"
    instance_folder = "Stowage-Planning-Benckmark/container_instances"

    
    results: List[ResultEntry] = []

    
    v_files = [Path(r) / f for r, _, fs in os.walk(vessel_folder)
               for f in fs if f.endswith(".txt")]

    i_files = [Path(r) / f for r, _, fs in os.walk(instance_folder)
               for f in fs if f.endswith(".txt")]

    inputs: List[Tuple[List[Container], Vessel, str, str]] = []
    for v_path in v_files:
        vessel = parse_benchmark_vessel(v_path)
        def fil(i: Path):
            stem = v_path.stem.capitalize()
            stem = stem[:-1] + stem[-1].upper()
            return stem in i.parts

        filtered_ifiles = filter(fil, i_files)
        for i_path in filtered_ifiles:
            conts = parse_benchmark_containers(i_path)

            inputs.append((conts, vessel, v_path.parts[-1], i_path.parts[-1]))

    sorted_inputs = sorted(
        inputs, key=lambda inp: len(inp[0]) * inp[1].capacity)
    
    count = len(sorted_inputs)

    for idx, inp in enumerate(sorted_inputs):
        executions = EXECUTIONS
        conts, vessel, v, i = inp
        if vessel.capacity < len(conts):
            print(
                f"[{idx+1}/{count}] Skippng: {v} + {i} -> {len(conts)} < {vessel.bays} * {vessel.rows} * {vessel.tiers} ({vessel.capacity})")
            continue
        random.shuffle(sols)
        for sol in sols:
            print(
                    f"[{idx+1}/{count}] Running {sol}: {v} + {i} -> ({len(conts)}, {vessel.bays} * {vessel.rows} * {vessel.tiers}):")

            budget = BUDGET_SECONDS
            elapsed = 0
            cost = 0
            executions -= skip[sol][(len(conts),
                                     vessel.bays, vessel.rows, vessel.tiers)]

            cur_exes = executions

            exes = 0
            for exe in range(executions):
                exes += 1
                start = time.perf_counter()

                cost += solutions[sol](conts, vessel, exe)

                el = time.perf_counter() - start
                elapsed += el
                budget -= el
                 
                if budget < 0:
                    print("\tBudget Skip")
                    break

            if exes:
                elapsed /= exes
                cost /= exes

                results.append({
                    "algo": sol,
                    "time": elapsed,
                    "cost": cost,
                    "config": f"{v} | {i}"
                })

    
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
