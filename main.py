from common import *
from exact import *
from genetic import *

# fazer test cases

# 2 X 5 X 6 = 60 - 5
# 2 X 5 X 6 = 60 - 10
# 2 X 5 X 6 = 60 - 15
# 2 X 5 X 6 = 60 - 20
# 2 X 5 X 6 = 60 - 25
# 2 X 5 X 6 = 60 - 30

# 10 X 5 X 5 = 250 - 50
# 10 X 5 X 5 = 250 - 100
# 10 X 5 X 5 = 250 - 150
# 10 X 5 X 5 = 250 - 200
# 10 X 5 X 5 = 250  - 250

# 10 X 10 X 10 = 1000 - 1000


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

    num_bays = 20
    num_rows = 10
    max_tiers = 5

    # You can change ALPHA/BETA to tune priority between rehandles and balance
    res = solve_stowage_3d_min_rehandles_with_cog(
        ContainerRamdom(60), num_bays, num_rows, max_tiers,
        ALPHA=1.0, BETA=0.05
    )

    print(res)


main()
