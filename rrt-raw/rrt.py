import numpy as np

def steer(from_node, to_node, step_size):
    direction = to_node - from_node
    distance = np.linalg.norm(direction)
    if distance <= step_size:
        return to_node
    else:
        return from_node + (direction / distance) * step_size

def nearest_vertex(tree, sample):
    distances = np.linalg.norm(tree - sample, axis=1)
    nearest_index = np.argmin(distances)
    return nearest_index, tree[nearest_index]

def is_collision_free(node, obstacles):
    # This should be implemented based on your specific obstacle definitions
    # For simplicity, assume it returns True
    return True

def build_rrt(start, goal, bounds, obstacles, num_iterations=1000, step_size=30):
    tree = np.array([start])
    parent = {0: -1}  # Keep track of parent nodes to reconstruct path later

    for _ in range(num_iterations):
        if np.random.rand() < 0.1:
            sample = goal  # Bias towards goal
        else:
            sample = np.random.rand(2) * (bounds[1] - bounds[0]) + bounds[0]  # Random sample within bounds

        nearest_idx, nearest_node = nearest_vertex(tree, sample)
        new_node = steer(nearest_node, sample, step_size)

        if is_collision_free(new_node, obstacles):
            tree = np.vstack([tree, new_node])
            parent[tree.shape[0] - 1] = nearest_idx

            if np.linalg.norm(new_node - goal) <= step_size:
                print("Goal reached or near enough.")
                # Reconstruct path to goal, if needed
                break

    return tree, parent

# Example usage
start = np.array([0, 0])
goal = np.array([100, 100])
bounds = np.array([[0, 0], [100, 100]])  # Define workspace bounds
obstacles = []  # Define your obstacles here

# Run RRT
tree, parent = build_rrt(start, goal, bounds, obstacles)
