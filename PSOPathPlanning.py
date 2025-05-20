import random
import matplotlib.pyplot as plt
import numpy as np

# Node class represents each point in the graph
class Node:
    def __init__(self, name, position=None):
        self.name = name
        self.adjacents = {}  # Dictionary of neighbor nodes and edge weights
        self.position = position  # (row, col) coordinates for heuristic calculation

    def add_neighbor(self, neighbor, weight):
        self.adjacents[neighbor] = weight

    def __lt__(self, other):
        return self.name < other.name

# Graph class to manage the network of nodes
class Graph:
    def __init__(self):
        self.nodes = {}

    def add_node(self, name, position=None):
        if name not in self.nodes:
            self.nodes[name] = Node(name, position)

    def add_edge(self, from_node, to_node, weight):
        self.add_node(from_node)
        self.add_node(to_node)
        self.nodes[from_node].add_neighbor(self.nodes[to_node], weight)
        self.nodes[to_node].add_neighbor(self.nodes[from_node], weight)  # Undirected graph

    def get_neighbors(self, node_name):
        return self.nodes[node_name].adjacents.items()

    def get_position(self, node_name):
        return self.nodes[node_name].position

# Particle class for PSO
class Particle:
    def __init__(self, graph, coverage_list):
        self.graph = graph
        self.coverage_list = coverage_list  # List of all grid cells to cover
        self.position = self.random_permutation()
        self.best_position = list(self.position)
        self.best_cost = self.evaluate(self.position)

    def random_permutation(self):
        perm = list(self.coverage_list)
        random.shuffle(perm)
        return perm

    def evaluate(self, path):
        cost = 0
        for i in range(len(path) - 1):
            for neighbor, weight in self.graph.get_neighbors(path[i]):
                if neighbor.name == path[i + 1]:
                    cost += weight
                    break
                else:
                    cost += 1.5  # slight penalty for indirect neighbors
        return cost

    def update(self, global_best):
        new_path = list(self.position)
        if random.random() < 0.5:
            idx = random.randint(0, len(new_path) - 2)
            new_path[idx], new_path[idx + 1] = new_path[idx + 1], new_path[idx]  # Swap
        else:
            split = random.randint(1, len(global_best) - 1)
            new_path = global_best[:split] + [n for n in self.position if n not in global_best[:split]]

        new_cost = self.evaluate(new_path)
        if new_cost < self.best_cost:
            self.best_cost = new_cost
            self.best_position = new_path
        self.position = new_path

# PSO algorithm for coverage path planning
def pso_coverage_planning(graph, cells_to_cover, num_particles=40, iterations=200):
    swarm = [Particle(graph, cells_to_cover) for _ in range(num_particles)]
    global_best = min(swarm, key=lambda p: p.best_cost).best_position
    global_best_cost = min(p.best_cost for p in swarm)

    for _ in range(iterations):
        for particle in swarm:
            particle.update(global_best)
        current_best = min(swarm, key=lambda p: p.best_cost)
        if current_best.best_cost < global_best_cost:
            global_best = list(current_best.best_position)
            global_best_cost = current_best.best_cost

    return global_best, global_best_cost

# Updated visualization to show full yard coverage
def visualize_coverage_path(graph, path):
    grid_size = 5
    grid = np.zeros((grid_size, grid_size))
    plt.figure(figsize=(6, 6))

    # Draw yard grid
    for node in graph.nodes.values():
        x, y = node.position[1] - 1, grid_size - node.position[0]
        plt.scatter(x, y, c='green', s=100)
        plt.text(x, y, node.name, fontsize=6, ha='center', va='center', color='white', bbox=dict(facecolor='black', edgecolor='none', boxstyle='round,pad=0.2'))

    # Draw coverage path
    for i in range(len(path) - 1):
        x1, y1 = graph.get_position(path[i])[1] - 1, grid_size - graph.get_position(path[i])[0]
        x2, y2 = graph.get_position(path[i + 1])[1] - 1, grid_size - graph.get_position(path[i + 1])[0]
        plt.plot([x1, x2], [y1, y2], color='blue', linewidth=2)

    plt.title("Robot Lawnmower Full Yard Coverage with PSO")
    plt.axis('equal')
    plt.xticks(range(grid_size))
    plt.yticks(range(grid_size))
    plt.grid(True)
    plt.show()

# Simulated scenario: Robot lawnmower full yard coverage with PSO
if __name__ == "__main__":
    graph = Graph()

    # Represent a 5x5 yard grid as nodes (R1C1 = Row 1, Column 1)
    all_cells = []
    for row in range(1, 6):
        for col in range(1, 6):
            name = f"R{row}C{col}"
            graph.add_node(name, (row, col))
            all_cells.append(name)

    # Connect adjacent yard cells
    for row in range(1, 6):
        for col in range(1, 6):
            current = f"R{row}C{col}"
            if col < 5:
                right = f"R{row}C{col+1}"
                graph.add_edge(current, right, 1)
            if row < 5:
                down = f"R{row+1}C{col}"
                graph.add_edge(current, down, 1)

    path_result, total_cost = pso_coverage_planning(graph, all_cells)
    print(f"PSO Robot lawnmower full coverage path: {path_result} with total cost: {total_cost}")

    visualize_coverage_path(graph, path_result)