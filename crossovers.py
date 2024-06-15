import random
import numpy as np



# AEX
def aex_crossover(parent1, parent2):
    num_cities = len(parent1)
    child = [None] * num_cities
    chosen = [False] * (num_cities + 1) 

    current_city = parent1[0]
    child[0] = current_city
    chosen[current_city] = True

    use_parent1 = False

    for i in range(1, num_cities):
        if use_parent1:
            next_city = next_city_from_parent(current_city, parent1)
        else:
            next_city = next_city_from_parent(current_city, parent2)

        if next_city is None or chosen[next_city]:
            remaining_cities = [city for city in parent1 if not chosen[city]]
            if not remaining_cities:
                break
            next_city = random.choice(remaining_cities)

        child[i] = next_city
        chosen[next_city] = True
        current_city = next_city
        use_parent1 = not use_parent1


    return child



def next_city_from_parent(current_city, parent):
    try:
        index = parent.index(current_city)
    except ValueError:
        return None

    next_index = (index + 1) % len(parent)
    next_city = parent[next_index]

    # Return the next city only if it's not visited yet in the current segment
    if next_city in parent[:index + 1]:
        return None

    return next_city



# OX1
def ordered_crossover(parent1, parent2):
    start, end = sorted([random.randint(0, len(parent1)-1) for _ in range(2)])
    # print(f"START: {start}")
    # print(f"END: {end}")
    child = [None] * len(parent1)

    child[start:end] = parent1[start:end]

    fill_positions = [i for i in range(len(parent2)) if parent2[i] not in child]
    # print(f"FILL POSITIONS: {fill_positions}")
    fill_index = 0

    for i in range(len(child)):
        if child[i] is None:
            child[i] = parent2[fill_positions[fill_index]]
            fill_index += 1

    return child


# PMX
def pmx_crossover(parent1, parent2):
    size = len(parent1)
    child = [None]*size

    start, end = sorted(random.sample(range(size), 2))

    child[start:end] = parent1[start:end]

    # Create a mapping from the section copied from the first parent
    mapping = {parent1[i]: parent2[i] for i in range(start, end)}

    # Fill in the remaining positions with elements from the second parent
    for i in list(range(0, start)) + list(range(end, size)):
        candidate = parent2[i]
        seen = set()
        while candidate in child:
            if candidate in seen:
                break 
            seen.add(candidate)
            candidate = mapping.get(candidate, candidate)

        child[i] = candidate

    return child


# ERX
def create_neighbor_list(parent1, parent2):
    size = len(parent1)
    neighbors = {key: set() for key in range(size)}

    for p in [parent1, parent2]:
        for i in range(size):
            left = p[i - 1]
            right = p[(i + 1) % size]
            neighbors[p[i]].update([left, right])

    return neighbors



def erx_crossover(parent1, parent2):
    neighbors = create_neighbor_list(parent1, parent2)
    # print(f"Neighbors: {neighbors}")
    size = len(parent1)

    current = np.random.choice(parent1)
    child = [current]

    # Remove the current vertex from all neighbor sets
    for nset in neighbors.values():
        nset.discard(current)

    while len(child) < size:
        if neighbors[current]:
            # Choose the neighbor with the shortest neighbor list
            next_vertex = min(neighbors[current], key=lambda x: len(neighbors[x]))
            neighbors[current].remove(next_vertex)
        else:
            # If no neighbors are left, pick randomly from the remaining vertices
            remaining_vertices = list(set(parent1) - set(child))
            next_vertex = np.random.choice(remaining_vertices)

        child.append(next_vertex)
        current = next_vertex

        # Remove the current vertex from all neighbor sets
        for nset in neighbors.values():
            nset.discard(current)

    return child


# CX
def cx_crossover(parent1, parent2):
    size = len(parent1)
    child1 = [None] * size
    child2 = [None] * size

    visited = [False] * size
    index_map = {value: idx for idx, value in enumerate(parent1)}
    
    if not visited[0]:
        current = 0
        cycle = []
        cycle_detected = False

        # Follow the cycle
        while not visited[current]:
            cycle.append(current)
            visited[current] = True
            current = index_map[parent2[current]]

            if current == 0:
                cycle_detected = True
                break

        # If cycle is detected, apply alternate pattern in filling child chromosomes
        if cycle_detected:
            for idx in cycle:
                child1[idx] = parent1[idx] if (len(cycle) % 2 == 1) else parent2[idx]
                child2[idx] = parent2[idx] if (len(cycle) % 2 == 1) else parent1[idx]

        # Handle case where we complete a cycle and check remaining unvisited vertices
        if not all(visited):
            unvisited = [i for i, v in enumerate(visited) if not v]
            for idx in unvisited:
                child1[idx] = parent2[idx]
                child2[idx] = parent1[idx]

    return child1, child2