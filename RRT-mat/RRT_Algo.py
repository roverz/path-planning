import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

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
    # we are ploting the ticks with interval of 100 from 0 to 1000
    plt.xticks(np.arange(0, 1001, 100))
    plt.yticks(np.arange(0, 1001, 100))
    plt.xlabel('X')
    plt.ylabel('Y')
    # to plot the grid path of background color gray and line style '--'
    plt.grid(True, color='gray', linestyle='--')
    plt.gca().set_aspect('equal', adjustable='box')
    # it creates a legend box typically on the upper right corner of the plot by default
    plt.legend()

def is_collision_free(new_node):
    for square in green_squares:
        # it is check that new_node coordinates falls within a specific range around the center of the green square
        # square[0]-50 -> represent the left boundary of the square, 50 units to the left of the square's center
        # square[0]+50 -> represents the right boundary of the square, 50 units to the right of the node squar's center
        # and square[1]-50 and square[1]+50 repersent the y-co-ordinates below and above
        if (square[0] - 60 <= new_node[0] <= square[0] + 60) and (square[1] - 60 <= new_node[1] <= square[1] + 60):
            return False
    # for square in yellow_squares:
    #     if (square[0] - 35 <= new_node[0] <= square[0] + 35) and (square[1] - 35 <= new_node[1] <= square[1] + 35):
    #         return False
    # if np.linalg.norm(new_node - np.array(red_cylinder) - np.array([50, 50])) <= 30:
    #     return False
    return True
# this function finds the nearest vertex in a given tree sturcture to a given sample point
def nearest_vertex(tree, sample):
    # this finds the euclidian distance
    distances = np.linalg.norm(tree - sample, axis=1)
    # np.argmin is a NumPy function that returns the indices of the minimum values along a specified axis.
    nearest_index = np.argmin(distances)
    # : the index of the nearest vertex (nearest_index) and the coordinates of that vertex in the tree.
    return nearest_index, tree[nearest_index]
# The steer function is used in the Rapidly-exploring Random Tree (RRT) algorithm to generate a new node 
def steer(from_node, to_node, step_size):
    # here form_node is nearest node and to_node is the sampled node so we need to find the direction in which we will move
    direction = to_node - from_node
    # the distance which we want to move along this direction vector
    length = np.linalg.norm(direction)
    direction_unit = direction / length
    step_length = min(step_size, length)
    new_node = from_node + direction_unit * step_length
    return new_node

def build_rrt(start, goal, num_iterations=1000, step_size=30):
    tree = np.array([start])
    parent_map = {0: -1}
    print("Node coordinates:")  # Initial message to indicate the start of coordinates output
    print(start)  # Print start node coordinates
    for i in range(num_iterations):
        sample = np.random.rand(2) * 1000
        if np.random.rand() < 0.1:
            sample = goal  # Directly aim for the goal with a 10% chance
        nearest_index, nearest_node = nearest_vertex(tree, sample)
        new_node = steer(nearest_node, sample, step_size)
        if is_collision_free(new_node):
            # Appends the new node to the tree.
            tree = np.vstack([tree, new_node]) #appending the new_node verically in the tree
            parent_map[tree.shape[0] - 1] = nearest_index #now setting the parent of the new node 
            print(new_node)  # Print the new node coordinates as they are added
            if np.linalg.norm(new_node - goal) <= step_size:
                print("Goal reached.")
                return tree, parent_map
    return tree, parent_map


# Run the algorithm and plot the environment and RRT
plot_environment()
tree, parent_map = build_rrt(start, goal, num_iterations=1000, step_size=50)

for i in range(1, tree.shape[0]):
    start_point = tree[parent_map[i]]
    end_point = tree[i]
    plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'r-')

plt.savefig('grid_with_rrt_path.png')
plt.show()
