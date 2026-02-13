import os
import pulp
from typing import List

from common import (
    Vessel, Container, Slot, SlotCoord,
    calculate_cost,
    W_REHANDLE, W_BALANCE, W_GM_FAIL, W_LEFTOVER,
)


def solve_stowage_lp(
    containers: List[Container],
    vessel: Vessel,
    ALPHA: float = W_REHANDLE,  # rehandle penalty — aligns with common.py weights
    BETA: float = W_BALANCE,    # CoG deviation penalty — aligns with common.py weights
) -> tuple[Vessel, float]:
    """
    ILP stowage solver.

    Parameters
    ----------
    vessel     : Empty Vessel (defines bay/row/tier grid and physics data).
    containers : Containers to stow (common.py Container dataclass).
    ALPHA      : Penalty per rehandle in the LP objective.
                 Defaults to W_REHANDLE from common.py.
    BETA       : Penalty per unit of CoG moment deviation in the LP objective.
                 Defaults to W_BALANCE from common.py.

    Returns
    -------
    (best_vessel, cost)
        best_vessel : Vessel with containers placed (or the empty vessel if
                      the solver found no feasible solution).
        cost        : calculate_cost() score from common.py — the canonical
                      objective combining GM safety, rehandles, balance, and
                      leftovers. This is what all solvers should be compared on.

    Notes on the LP objective vs. calculate_cost()
    -----------------------------------------------
    calculate_cost() uses a numpy hydrostatic interpolation (non-linear) for GM,
    which cannot be embedded directly in a linear programme. The LP therefore
    uses a tractable linear proxy:
      - ALPHA * rehandles  (exact rehandle count)
      - BETA  * (|bay_moment_dev| + |row_moment_dev| + |tier_moment_dev|)
    After solving, the canonical calculate_cost() score is computed on the
    resulting Vessel so results are comparable across all solvers.
    """

    B = list(range(vessel.bays))
    R = list(range(vessel.rows))
    T = list(range(1, vessel.tiers + 1))   # tiers numbered 1..max_tiers
    C = list(range(len(containers)))

    dest = {i: containers[i].dischargePort for i in C}
    weight = {i: containers[i].weight for i in C}
    total_weight = sum(weight[i] for i in C)

    target_bay = (vessel.bays - 1) / 2.0
    target_row = (vessel.rows - 1) / 2.0
    target_tier = 0.0

    prob = pulp.LpProblem("Stowage_LP", pulp.LpMinimize)

    # ------------------------------------------------------------------
    # Decision variables
    # ------------------------------------------------------------------

    # x[b][r][t][i] = 1  iff container i is placed at (bay b, row r, tier t)
    x = pulp.LpVariable.dicts(
        "x", (B, R, T, C), lowBound=0, upBound=1, cat="Binary"
    )

    # y[(i,j)][b][r] = 1  iff container i (earlier dest) is below j (later dest)
    # in the same column — i.e. a rehandle
    pairs = [(i, j) for i in C for j in C if dest[i] < dest[j]]
    y = pulp.LpVariable.dicts(
        "y", (pairs, B, R), lowBound=0, upBound=1, cat="Binary"
    )

    # Absolute deviation variables for the three CoG axes (linearised |·|)
    Db_pos = pulp.LpVariable("Db_pos", lowBound=0)
    Db_neg = pulp.LpVariable("Db_neg", lowBound=0)
    Dr_pos = pulp.LpVariable("Dr_pos", lowBound=0)
    Dr_neg = pulp.LpVariable("Dr_neg", lowBound=0)
    Dt_pos = pulp.LpVariable("Dt_pos", lowBound=0)
    Dt_neg = pulp.LpVariable("Dt_neg", lowBound=0)

    # ------------------------------------------------------------------
    # Objective: linear proxy for calculate_cost()
    # ------------------------------------------------------------------
    sum_rehandles = pulp.lpSum(
        y[(i, j)][b][r] for (i, j) in pairs for b in B for r in R
    )
    sum_cog_dev = Db_pos + Db_neg + Dr_pos + Dr_neg + Dt_pos + Dt_neg

    prob += ALPHA * sum_rehandles + BETA * sum_cog_dev

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    # 1. Each container placed exactly once
    for i in C:
        prob += (
            pulp.lpSum(x[b][r][t][i] for b in B for r in R for t in T) == 1,
            f"place_once_{i}",
        )

    # 2. At most one container per slot
    for b in B:
        for r in R:
            for t in T:
                prob += (
                    pulp.lpSum(x[b][r][t][i] for i in C) <= 1,
                    f"slot_unique_b{b}_r{r}_t{t}",
                )

    # 3. No floating containers: tier t occupied only if tier t-1 occupied
    for b in B:
        for r in R:
            for t in T:
                if t == 1:
                    continue
                prob += (
                    pulp.lpSum(x[b][r][t][i] for i in C)
                    <= pulp.lpSum(x[b][r][t - 1][i] for i in C),
                    f"no_float_b{b}_r{r}_t{t}",
                )

    # 4. Rehandle linking: y[(i,j)][b][r] is 1 when i sits below j in (b,r)
    for (i, j) in pairs:
        for b in B:
            for r in R:
                for t1 in T:
                    for t2 in T:
                        if t1 < t2:
                            prob += (
                                y[(i, j)][b][r]
                                >= x[b][r][t1][i] + x[b][r][t2][j] - 1,
                                f"rh_link_i{i}_j{j}_b{b}_r{r}_t{t1}_{t2}",
                            )
                # Upper bounds: y can only be 1 if both i and j are in column (b,r)
                prob += (
                    y[(i, j)][b][r]
                    <= pulp.lpSum(x[b][r][t][i] for t in T),
                    f"rh_ub1_i{i}_j{j}_b{b}_r{r}",
                )
                prob += (
                    y[(i, j)][b][r]
                    <= pulp.lpSum(x[b][r][t][j] for t in T),
                    f"rh_ub2_i{i}_j{j}_b{b}_r{r}",
                )

    # 5. CoG moment balance (linearised absolute deviation)
    bay_moment = pulp.lpSum(
        weight[i] * b * x[b][r][t][i]
        for i in C for b in B for r in R for t in T
    )
    row_moment = pulp.lpSum(
        weight[i] * r * x[b][r][t][i]
        for i in C for b in B for r in R for t in T
    )
    tier_moment = pulp.lpSum(
        weight[i] * (t - 1) * x[b][r][t][i]
        for i in C for b in B for r in R for t in T
    )

    prob += bay_moment - target_bay * total_weight == Db_pos - Db_neg, "bay_balance"
    prob += row_moment - target_row * total_weight == Dr_pos - Dr_neg, "row_balance"
    prob += tier_moment - target_tier * \
        total_weight == Dt_pos - Dt_neg, "tier_balance"

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    solver = pulp.PULP_CBC_CMD(msg=True, threads=os.cpu_count())
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    print(f"[LP] Solver status: {status}")

    # ------------------------------------------------------------------
    # Extract solution into a Vessel
    # ------------------------------------------------------------------
    result_vessel = Vessel(
        bays=vessel.bays,
        rows=vessel.rows,
        tiers=vessel.tiers,
        lightship_weight=vessel.lightship_weight,
        lightship_vcg=vessel.lightship_vcg,
        hydro_disp=vessel.hydro_disp,
        hydro_km=vessel.hydro_km,
    )

    leftovers: List[Container] = []

    if prob.status == pulp.LpStatusOptimal or prob.status == 1:  # Optimal or feasible
        placed = set()
        for i in C:
            for b in B:
                for r in R:
                    for t in T:
                        if pulp.value(x[b][r][t][i]) is not None and round(pulp.value(x[b][r][t][i])) == 1:
                            # LP uses 1-indexed tiers; Vessel uses 0-indexed
                            coord = SlotCoord(b, r, t - 1)
                            slot = result_vessel.get_slot_at(coord)
                            if slot is not None:
                                result_vessel.place(containers[i], slot)
                                placed.add(i)
        leftovers = [containers[i] for i in C if i not in placed]
    else:
        print("[LP] No feasible solution found — returning empty vessel.")
        leftovers = list(containers)

    # ------------------------------------------------------------------
    # Canonical cost via common.py (used for cross-solver comparison)
    # ------------------------------------------------------------------
    cost = calculate_cost(result_vessel, leftovers)

    rehandles = result_vessel.calculate_rehandles()
    print(f"[LP] Rehandles: {rehandles}")
    print(f"[LP] Stowed: {result_vessel.containerAmount}/{len(containers)}")
    print(f"[LP] calculate_cost() score: {cost:.0f}")

    return result_vessel, cost
