import pulp
from common import *
from typing import List


def solve_stowage_3d_min_rehandles_with_cog(
    containers: List[Cont],
    num_bays: int,
    num_rows: int,
    max_tiers: int,
    ALPHA: float = 1.0,   # weight for rehandles (primary objective)
    BETA: float = 0.01    # weight for CoG deviation (secondary)
) -> Ship:
    """
    containers: list of dicts with keys 'id', 'weight', 'dest'
    returns dict containing: status, rehandles (int), weighted_objective (float),
        assignment {container_id: (bay,row,tier)}, rehandles_by_stack
    """

    # indices
    C = list(range(len(containers)))
    B = list(range(num_bays))
    R = list(range(num_rows))
    T = list(range(1, max_tiers + 1))   # tiers numbered 1..max_tiers

    dest = {i: containers[i]['dest'] for i in C}
    weight = {i: containers[i]['weight'] for i in C}
    total_weight = sum(weight[i] for i in C)

    target_bay = (num_bays - 1) / 2.0    # middle bay
    target_row = (num_rows - 1) / 2.0    # middle row
    target_tier = 0.0                     # bottom

    prob = pulp.LpProblem("Stowage3D_MinRehandles_with_CoG", pulp.LpMinimize)

    # Decision variables: x[b][r][t][i] (note ordering to allow x[b][r][t][i] access)
    x = pulp.LpVariable.dicts(
        "x", (B, R, T, C), lowBound=0, upBound=1, cat="Binary")

    # Rehandle pairs: only pairs where dest[i] < dest[j]
    pairs = [(i, j) for i in C for j in C if dest[i] < dest[j]]
    y = pulp.LpVariable.dicts(
        "y", (pairs, B, R), lowBound=0, upBound=1, cat="Binary")

    Db_pos = pulp.LpVariable("Db_pos", lowBound=0)
    Db_neg = pulp.LpVariable("Db_neg", lowBound=0)
    Dr_pos = pulp.LpVariable("Dr_pos", lowBound=0)
    Dr_neg = pulp.LpVariable("Dr_neg", lowBound=0)
    Dt_pos = pulp.LpVariable("Dt_pos", lowBound=0)
    Dt_neg = pulp.LpVariable("Dt_neg", lowBound=0)

    sum_rehandles = pulp.lpSum(y[(i, j)][b][r_]
                               for (i, j) in pairs for b in B for r_ in R)
    sum_cog_dev = Db_pos + Db_neg + Dr_pos + Dr_neg + Dt_pos + Dt_neg
    prob += ALPHA * sum_rehandles + BETA * sum_cog_dev

    # 1) Each container placed exactly once
    for i in C:
        prob += pulp.lpSum(x[b][r][t][i]
                           for b in B for r in R for t in T) == 1, f"place_once_{i}"

    # 2) One container per slot
    for b in B:
        for r in R:
            for t in T:
                prob += pulp.lpSum(x[b][r][t][i]
                                   for i in C) <= 1, f"slot_unique_b{b}_r{r}_t{t}"

    # 3) No floating containers (tier t occupied only if tier t-1 occupied)
    for b in B:
        for r in R:
            for t in T:
                if t == 1:
                    continue
                prob += pulp.lpSum(x[b][r][t][i] for i in C) <= \
                    pulp.lpSum(x[b][r][t-1][i]
                               for i in C), f"no_float_b{b}_r{r}_t{t}"

    # 4) Rehandling definition: r[(i,j)][b][r_] >= x[b][r_][t1][i] + x[b][r_][t2][j] - 1  for t1 < t2
    for (i, j) in pairs:
        for b in B:
            for r in R:
                for t1 in T:
                    for t2 in T:
                        if t1 < t2:
                            prob += y[(i, j)][b][r] >= x[b][r][t1][i] + x[b][r][t2][j] - 1, \
                                f"r_link_i{i}_j{j}_b{b}_r{r}_t{t1}_{t2}"
                prob += y[(i, j)][b][r] <= pulp.lpSum(x[b][r][t][i]
                                                      for t in T), f"r_ub1_i{i}_j{j}_b{b}_r{r}"
                prob += y[(i, j)][b][r] <= pulp.lpSum(x[b][r][t][j]
                                                      for t in T), f"r_ub2_i{i}_j{j}_b{b}_r{r}"

    bay_moment = pulp.lpSum(weight[i] * b * x[b][r][t][i]
                            for i in C for b in B for r in R for t in T)
    row_moment = pulp.lpSum(weight[i] * r * x[b][r][t][i]
                            for i in C for b in B for r in R for t in T)
    tier_moment = pulp.lpSum(weight[i] * (t - 1) * x[b][r][t][i]
                             for i in C for b in B for r in R for t in T)

    target_bay_moment = target_bay * total_weight
    target_row_moment = target_row * total_weight
    target_tier_moment = target_tier * total_weight

    prob += bay_moment - target_bay_moment == Db_pos - Db_neg, "bay_cog_balance"
    prob += row_moment - target_row_moment == Dr_pos - Dr_neg, "row_cog_balance"
    prob += tier_moment - target_tier_moment == Dt_pos - Dt_neg, "tier_cog_balance"

    solver = pulp.PULP_CBC_CMD(msg=True)
    result = prob.solve(solver)

    status_str = pulp.LpStatus[prob.status]

    ship: Ship = [[[None for _ in T] for _ in R] for _ in B]

    for i in C:
        for b in B:
            for r_ in R:
                for t in T:
                    if pulp.value(x[b][r_][t][i]) == 1:
                        ship[b][r_][t-1] = containers[i]

    return ship
