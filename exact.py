# stowage_3d_rehandle_milp_with_cog.py
# 3D exact MILP for container stowage minimizing rehandles (reshuffles)
# + linearized Center-of-Gravity (CoG) balancing on bay, row and tier axes.
#
# Decision vars:
#   x[b][r][t][i]  = 1 if container i placed at bay b, row r, tier t
#   r[(i,j)][b][r] = 1 if container j blocks i in stack (b,r)
#
# Objective: ALPHA * (#rehandles) + BETA * (sum of CoG deviations)
#
# Tier is modeled so lower tiers correspond to smaller vertical position values;
# target_tier = 0 (prefer as low as possible).

import pulp
from common import *
from typing import List, Dict, Optional

def solve_stowage_3d_min_rehandles_with_cog(
    containers: List[Dict],
    num_bays: int,
    num_rows: int,
    max_tiers: int,
    ALPHA: float = 1.0,   # weight for rehandles (primary objective)
    BETA: float = 0.01    # weight for CoG deviation (secondary)
) -> Dict:
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

    id_of = {i: containers[i]['id'] for i in C}
    dest  = {i: containers[i]['dest'] for i in C}
    weight = {i: containers[i]['weight'] for i in C}

    # total weight (constant)
    total_weight = sum(weight[i] for i in C)

    # Target CoG positions
    target_bay  = (num_bays - 1) / 2.0    # middle bay (0-indexed)
    target_row  = (num_rows - 1) / 2.0    # middle row
    target_tier = 0.0                     # "low" target: prefer low tiers (tier 1 -> pos 0)

    prob = pulp.LpProblem("Stowage3D_MinRehandles_with_CoG", pulp.LpMinimize)

    # Decision variables: x[b][r][t][i] (note ordering to allow x[b][r][t][i] access)
    x = pulp.LpVariable.dicts("x", (B, R, T, C), lowBound=0, upBound=1, cat="Binary")

    # Rehandle pairs: only pairs where dest[i] < dest[j]
    pairs = [(i, j) for i in C for j in C if dest[i] < dest[j]]
    rvar = pulp.LpVariable.dicts("r", (pairs, B, R), lowBound=0, upBound=1, cat="Binary")

    # CoG deviation variables (non-negative)
    Db_pos = pulp.LpVariable("Db_pos", lowBound=0)
    Db_neg = pulp.LpVariable("Db_neg", lowBound=0)
    Dr_pos = pulp.LpVariable("Dr_pos", lowBound=0)
    Dr_neg = pulp.LpVariable("Dr_neg", lowBound=0)
    Dt_pos = pulp.LpVariable("Dt_pos", lowBound=0)
    Dt_neg = pulp.LpVariable("Dt_neg", lowBound=0)

    # -------------------------
    # Objective: weighted sum
    # -------------------------
    # rehandles sum:
    sum_rehandles = pulp.lpSum(rvar[(i, j)][b][r_] for (i, j) in pairs for b in B for r_ in R)

    # CoG deviations (we'll add constraints to link moments to these vars)
    sum_cog_dev = Db_pos + Db_neg + Dr_pos + Dr_neg + Dt_pos + Dt_neg

    prob += ALPHA * sum_rehandles + BETA * sum_cog_dev

    # -------------------------
    # Constraints
    # -------------------------

    # 1) Each container placed exactly once
    for i in C:
        prob += pulp.lpSum(x[b][r_][t][i] for b in B for r_ in R for t in T) == 1, f"place_once_{i}"

    # 2) One container per slot
    for b in B:
        for r_ in R:
            for t in T:
                prob += pulp.lpSum(x[b][r_][t][i] for i in C) <= 1, f"slot_unique_b{b}_r{r_}_t{t}"

    # 3) No floating containers (tier t occupied only if tier t-1 occupied)
    for b in B:
        for r_ in R:
            for t in T:
                if t == 1:
                    continue
                prob += pulp.lpSum(x[b][r_][t][i] for i in C) <= \
                        pulp.lpSum(x[b][r_][t-1][i] for i in C), f"no_float_b{b}_r{r_}_t{t}"

    # 4) Rehandling definition: r[(i,j)][b][r_] >= x[b][r_][t1][i] + x[b][r_][t2][j] - 1  for t1 < t2
    for (i, j) in pairs:
        for b in B:
            for r_ in R:
                # lower bound linking constraints
                for t1 in T:
                    for t2 in T:
                        if t1 < t2:
                            prob += rvar[(i, j)][b][r_] >= x[b][r_][t1][i] + x[b][r_][t2][j] - 1, \
                                    f"r_link_i{i}_j{j}_b{b}_r{r_}_t{t1}_{t2}"
                # optional upper bounds to tighten rvar (r <= occupancy sums)
                prob += rvar[(i, j)][b][r_] <= pulp.lpSum(x[b][r_][t][i] for t in T), f"r_ub1_i{i}_j{j}_b{b}_r{r_}"
                prob += rvar[(i, j)][b][r_] <= pulp.lpSum(x[b][r_][t][j] for t in T), f"r_ub2_i{i}_j{j}_b{b}_r{r_}"

    # -------------------------
    # CoG (moment) definitions and linking to deviation vars
    # -------------------------
    # bay moment: sum_i sum_{b,r,t} weight[i] * b * x[b][r][t][i]
    bay_moment = pulp.lpSum(weight[i] * b * x[b][r_][t][i]
                             for i in C for b in B for r_ in R for t in T)

    # row moment
    row_moment = pulp.lpSum(weight[i] * r_ * x[b][r_][t][i]
                             for i in C for b in B for r_ in R for t in T)

    # tier moment: use (t - 1) so tier 1 -> position 0 (lowest)
    tier_moment = pulp.lpSum(weight[i] * (t - 1) * x[b][r_][t][i]
                              for i in C for b in B for r_ in R for t in T)

    # target moments
    target_bay_moment = target_bay * total_weight
    target_row_moment = target_row * total_weight
    target_tier_moment = target_tier * total_weight

    # linking equalities: moment - target = pos - neg
    prob += bay_moment - target_bay_moment == Db_pos - Db_neg, "bay_cog_balance"
    prob += row_moment - target_row_moment == Dr_pos - Dr_neg, "row_cog_balance"
    prob += tier_moment - target_tier_moment == Dt_pos - Dt_neg, "tier_cog_balance"


    solver = pulp.PULP_CBC_CMD(msg=True)
    result = prob.solve(solver)

    status_str = pulp.LpStatus[prob.status]

    # assignment = {}
    # for i in C:
    #     pos = None
    #     for b in B:
    #         for r_ in R:
    #             for t in T:
    #                 val = pulp.value(x[b][r_][t][i])
    #                 if val is not None and val > 0.5:
    #                     pos = (b, r_, t)
    #                     break
    #             if pos is not None:
    #                 break
    #         if pos is not None:
    #             break
    #     assignment[id_of[i]] = pos
    
    ship = [[[None for _ in T] for _ in R] for _ in B]

    for i in C:
        for b in B:
            for r_ in R:
                for t in T:
                    if pulp.value(x[b][r_][t][i]) == 1:
                        ship[b][r_][t-1] = containers[i]
        

    return {
        "rehandles": RehandlesNumber( ship ),
        "bay_moment": BayMoment(ship, num_bays ),
        "row_moment": RowMoment(ship, num_rows ),
        "tier_moment": TierMoment(ship),
    }

