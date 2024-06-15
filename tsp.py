import random
import math
import matplotlib.pyplot as plt
import numpy as np
import time
from crossovers import aex_crossover, pmx_crossover, erx_crossover, ordered_crossover, cx_crossover

# Get cities information in a list of lists, where each line in the text file is a list of types [str, int, int]
def getCity(txt_file):
    cities = []
    f = open(txt_file)
    for i in f.readlines():
        node_city_val = i.split()
        cities.append(
            [node_city_val[0], float(node_city_val[1]), float(node_city_val[2])]
        )

    return cities


def compute_distance_matrix(cities):
    num_cities = len(cities)
    distance_matrix = np.zeros((num_cities, num_cities), dtype=np.float16) # Change dtype depending on speed and/or precision needs
    
    for i in range(num_cities):
        for j in range(i, num_cities):
            distance = math.sqrt(
                math.pow(cities[j][1] - cities[i][1], 2) + math.pow(cities[j][2] - cities[i][2], 2)
            )
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance

    return distance_matrix


def calc_distance_matrix(route, distance_matrix):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i]][route[i + 1]]
    total_distance += distance_matrix[route[-1]][route[0]] 
    return total_distance


def selectPopulation(size, lenCities, distance_matrix):
    population = []

    for _ in range(size):
        route = list(range(lenCities))
        random.shuffle(route)
        # print(f"Route: {route}")
        distance = calc_distance_matrix(route, distance_matrix)
        # print(f"Distance: {distance}")
        population.append([distance, route])

    population.sort(key=lambda x: x[0])
    fittest = population[0]

    return population, fittest


def inversion_mutation(chromosome):
    start, end = sorted([random.randint(0, len(chromosome) - 1) for _ in range(2)])
    chromosome[start:end] = chromosome[start:end][::-1]
    return chromosome


def tournament_selection(population, tournament_size):
    selected = random.choices(population, k=tournament_size)
    return min(selected, key=lambda x: x[0])


def adaptive_mutation_rate(population, base_rate):
    distances = [ind[0] for ind in population]
    diversity = len(set(distances)) / len(population)
    return base_rate + (1 - diversity) * 0.1


def drawMap(cities, best_route, title="TSP Solution"):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    
    # Drawing each city as a point and annotating it
    for city in cities:
        plt.plot(city[1], city[2], "ro")
        plt.annotate(city[0], (city[1], city[2]))

    # Drawing lines between cities in the order of the best route
    for i in range(len(best_route) - 1):
        first_idx = best_route[i]
        second_idx = best_route[(i + 1)]
        first_city = cities[first_idx]
        second_city = cities[second_idx]
        
        # Draw a line between the two cities
        plt.plot([first_city[1], second_city[1]], [first_city[2], second_city[2]], "gray")
    
    # Plot from last to first city    
    plt.plot([cities[best_route[0]][1], cities[best_route[-1]][1]], [cities[best_route[0]][2], cities[best_route[-1]][2]], "gray")

    plt.show()
    

def crossover_wrapper(parent1, parent2, crossover_func):
    result = crossover_func(parent1, parent2)
    if isinstance(result, tuple):
        # Assuming the crossover returns a tuple of two children, for cx_crossover
        return list(result)
    else:
        # If only one child is returned, wrap it in a list
        return [result]


def genetic_algorithm(cities, num_generations=200, population_size=100, mutation_rate=0.1, tournament_size=5, crossover_func=aex_crossover):
    distance_matrix = compute_distance_matrix(cities)
    lenCities = len(cities)
    population, best_ever = selectPopulation(population_size, lenCities, distance_matrix)
    
    best_distances = [best_ever[0]]

    for generation in range(num_generations):
        new_population = []
        
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, tournament_size)
            parent2 = tournament_selection(population, tournament_size)
            
            children_routes = crossover_wrapper(parent1[1], parent2[1], crossover_func)

            for child_route in children_routes:
                if random.random() < adaptive_mutation_rate(population, mutation_rate):
                    child_route = inversion_mutation(child_route)

                child_distance = calc_distance_matrix(child_route, distance_matrix)
                new_population.append([child_distance, child_route])
                if len(new_population) >= population_size:
                    break

        new_population.sort(key=lambda x: x[0])
        population = new_population
        current_best = population[0]
        best_distances.append(current_best[0])
        
        if current_best[0] < best_ever[0]:
            best_ever = current_best
        
        # if generation % 10 == 0:
        print(f"Generation {generation} best distance = {best_ever[0]}")
            
        
    plt.plot(best_distances)
    plt.xlabel('Generation')
    plt.ylabel('Distance')
    plt.title(f'Best distance by generation for CX')
    plt.show()

    return best_ever

start_time = time.time()

cities_file = "16kut.txt"
cities = getCity(cities_file)

# Choose the crossover function to use
chosen_crossover = ordered_crossover  # aex_crossover, pmx_crossover, erx_crossover, ordered_crossover, cx_crossover

best_route = genetic_algorithm(cities, crossover_func=chosen_crossover)
print("Best route found:", best_route[1])
print("Distance of best route:", best_route[0])

# drawMap(cities=cities, best_route=best_route[1])

end_time = time.time()
print("Total execution time: {} seconds".format(end_time - start_time))