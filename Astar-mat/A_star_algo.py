import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import heapq

# Define obstacle positions and start/goal points
green_squares = [(400, 400), (300, 600), (600, 700), (900, 800), (200, 800)]
yellow_squares = [(100, 100), (300, 300), (500, 500), (700, 700), (250, 700), (700, 200)]
red_cylinder = (900, 0)
start = np.array([50, 950])
goal = np.array([950, 50])

def plot_environment():
    plt.figure(figsize=(8, 6))
    for square in green_squares:
        plt.gca().add_patch(Rectangle((square[0] - 50, square[1] - 50), 100, 100, color='green'))
    for square in yellow_squares:
        plt.gca().add_patch(Rectangle((square[0] - 25, square[1] - 25), 50, 50, color='yellow'))
    plt.gca().add_patch(Circle((red_cylinder[0] + 50, red_cylinder[1] + 50), 30, color='red'))
    plt.scatter(*start, color='blue', label='Start')
    plt.scatter(*goal, color='orange', label='Goal')
    plt.xlim(0, 1000)
    plt.ylim(0, 1000)
    plt.xticks(np.arange(0, 1001, 100))
    plt.yticks(np.arange(0, 1001, 100))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, color='gray', linestyle='--')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()

def is_collision_free(new_node):
    for square in green_squares:
        if (square[0] - 60 <= new_node[0] <= square[0] + 60) and (square[1] - 60 <= new_node[1] <= square[1] + 60):
            return False
    # for square in yellow_squares:
    #     if (square[0] - 35 <= new_node[0] <= square[0] + 35) and (square[1] - 35 <= new_node[1] <= square[1] + 35):
    #         return False
    # if np.linalg.norm(new_node - np.array(red_cylinder) - np.array([50, 50])) <= 30:
    #     return False
    return True

def nearest_vertex(tree, sample):
    distances = np.linalg.norm(tree - sample, axis=1)
    nearest_index = np.argmin(distances)
    return nearest_index, tree[nearest_index]

def steer(from_node, to_node, step_size):
    direction = to_node - from_node
    length = np.linalg.norm(direction)
    direction_unit = direction / length
    step_length = min(step_size, length)
    new_node = from_node + direction_unit * step_length
    return new_node

def build_rrt(start, goal, num_iterations=5000, step_size=30):  # Adjusted num_iterations to 5000
    tree = np.array([start])
    parent_map = {0: -1}
    for i in range(num_iterations):
        sample = np.random.rand(2) * 1000
        if np.random.rand() < 0.1:
            sample = goal
        nearest_index, nearest_node = nearest_vertex(tree, sample)
        new_node = steer(nearest_node, sample, step_size)
        if is_collision_free(new_node):
            tree = np.vstack([tree, new_node])
            parent_map[tree.shape[0] - 1] = nearest_index
            if np.linalg.norm(new_node - goal) <= step_size:
                print("Goal reached.")
                return tree, parent_map
    return tree, parent_map

def heuristic(a, b):
    return np.linalg.norm(a - b)

def a_star_search(tree, start_index, goal_index, parent_map):
    #  A priority queue that keeps track of nodes to be evaluated
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(tree[start_index], tree[goal_index]), start_index))
# map each node to its predecessor on the cheapest path from start to goal once that path has been found.
    came_from = {}
    # intilizing all node g_score as infinity(np.inf) and also f_score
    g_score = {node_index: np.inf for node_index in range(len(tree))}
    # for starting node the g_score will be 0
    g_score[start_index] = 0
    f_score = {node_index: np.inf for node_index in range(len(tree))}
    # and for starting index the f_score will be it's huiristic because f_score=g_score+h_score,
    # so for starting node g_score is 0 that's why f_score is h_score(huiristic)
    f_score[start_index] = heuristic(tree[start_index], tree[goal_index])

    while open_set:
        # now in the starting current is start node and we are extracting the index of the current node
        current = heapq.heappop(open_set)[1]
        if current == goal_index:
            return reconstruct_path(came_from, current)
        for neighbor in get_neighbors(tree, current, parent_map):
            tentative_g_score = g_score[current] + np.linalg.norm(tree[current] - tree[neighbor])
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(tree[neighbor], tree[goal_index])
                if all(neighbor != elem[1] for elem in open_set):
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None

def get_neighbors(tree, node_index, parent_map):
    neighbors = []
    for child, parent in parent_map.items():
        if parent == node_index:
            neighbors.append(child)
        elif child == node_index and parent != -1:
            neighbors.append(parent)
    return neighbors

def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.insert(0, current)
    return total_path

# Plotting the environment, RRT, and A* path
plot_environment()  # Plot the initial environment

# Build RRT
tree, parent_map = build_rrt(start, goal, num_iterations=1000, step_size=50)

# Plot RRT
for i in range(1, tree.shape[0]):
    start_point = tree[parent_map[i]]
    end_point = tree[i]
    plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'r-')

# Find closest node to goal
goal_index = np.argmin(np.linalg.norm(tree - goal, axis=1))

# A* search for the shortest path
path_indices = a_star_search(tree, 0, goal_index, parent_map)
# Print A* path if found
if path_indices:
    a_star_path = tree[path_indices]
    print("A* Shortest Path Coordinates:")
    for node in a_star_path:
        print(node)

# The expression a_star_path[:, 0] extracts all the x-coordinates of the nodes in the a_star_path,
# while a_star_path[:, 1] extracts all the y-coordinates of the nodes.
# Plot A* path if found
if path_indices:
    a_star_path = tree[path_indices]
    plt.plot(a_star_path[:, 0], a_star_path[:, 1], 'b-', linewidth=3, label='A* Path')

plt.legend()
plt.savefig('grid_with_rrt_and_astar_path.png')
plt.show()
