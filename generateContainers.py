import random
from typing import List

from common import Cont


def ContainerRamdom(number: int) -> List[Cont]:

    # Generating 250 containers starting from id 11
    containers_250: List[Cont] = [
        {
            'id': i,
            'weight': round(random.uniform(1.0, 100.0), 1),
            'dest': random.randint(1, 5)
        }
        for i in range(1, number + 1)
    ]

    return containers_250


def main():
    with open("containers/containers-5.txt", "w") as file:
        file.write(str(ContainerRamdom(5)))
    with open("containers/containers-10.txt", "w") as file:
        file.write(str(ContainerRamdom(10)))
    with open("containers/containers-15.txt", "w") as file:
        file.write(str(ContainerRamdom(15)))
    with open("containers/containers-20.txt", "w") as file:
        file.write(str(ContainerRamdom(20)))
    with open("containers/containers-25.txt", "w") as file:
        file.write(str(ContainerRamdom(25)))

    with open("containers/containers-50.txt", "w") as file:
        file.write(str(ContainerRamdom(50)))
    with open("containers/containers-100.txt", "w") as file:
        file.write(str(ContainerRamdom(100)))
    with open("containers/containers-150.txt", "w") as file:
        file.write(str(ContainerRamdom(150)))
    with open("containers/containers-200.txt", "w") as file:
        file.write(str(ContainerRamdom(200)))
    with open("containers/containers-250.txt", "w") as file:
        file.write(str(ContainerRamdom(250)))
    with open("containers/containers-1000.txt", "w") as file:
        file.write(str(ContainerRamdom(1000)))


main()
