import random
import math
import matplotlib.pyplot as plt
import numpy as np
import time
from crossovers import aex_crossover, pmx_crossover, erx_crossover, ordered_crossover, cx_crossover
import tsp


crossover_methods = {
    'AEX': aex_crossover,
    'PMX': pmx_crossover,
    'ERX': erx_crossover,
    'Ordered': ordered_crossover,
    'CX': cx_crossover
}

# Load cities from a file or define them
cities = tsp.getCity(txt_file="tsp_cities.txt")

for name, func in crossover_methods.items():
    print(f"Running GA with {name} crossover")
    best_route = tsp.genetic_algorithm(cities, crossover_func=func)
    print(f"Best route for {name}: {best_route[1]}")
    print(f"Distance of best route for {name}: {best_route[0]}")