import pygame
import numpy as np
from matplotlib.patches import Rectangle, Circle
import heapq
from scipy import interpolate
from scipy.interpolate import BSpline

# Initialize Pygame
pygame.init()

# Define window dimensions
window_width = 1000
window_height = 1000
scale_factor=1.0
scaled_window_width=int(window_width*scale_factor)
scaled_window_height=int(window_height*scale_factor)
window = pygame.display.set_mode((window_width, window_height))

# Load car image and scale it
car_img = pygame.image.load('tesla.png')
car_img = pygame.transform.scale(car_img, (30, 60))

# Define obstacle positions and start/goal points
green_squares = [(400, 400), (300, 600), (600, 700), (900, 800), (200, 800)]
yellow_squares = [(100, 100), (300, 300), (500, 500), (700, 700), (250, 700), (700, 200)]
red_cylinder = (900, 0)
start = np.array([50, 950])
goal = np.array([950, 50])


# Function to draw obstacles and start/goal points
def draw_environment():
    window.fill((255, 255, 255))  # Clear the window

    # Draw obstacles
    for square in green_squares:
        pygame.draw.rect(window, (0, 255, 0), (square[0] - 50, square[1] - 50, 100, 100))
    for square in yellow_squares:
        pygame.draw.rect(window, (255, 255, 0), (square[0] - 25, square[1] - 25, 50, 50))
    pygame.draw.circle(window, (255, 0, 0), (red_cylinder[0] + 50, red_cylinder[1] + 50), 30)

    # Draw start and goal points
    pygame.draw.circle(window, (0, 0, 255), start, 5)
    pygame.draw.circle(window, (255, 165, 0), goal, 5)

def handle_zoom(event):
    global scale_factor, scaled_window_width, scaled_window_height, window

    if event.key == pygame.K_EQUALS:  # Zoom in
        scale_factor += 0.1
    elif event.key == pygame.K_MINUS:  # Shrink
        scale_factor = max(0.2, scale_factor - 0.1)

    # Update scaled window dimensions
    scaled_window_width = int(window_width * scale_factor)
    scaled_window_height = int(window_height * scale_factor)

    # Recreate the Pygame window with updated size
    window = pygame.display.set_mode((scaled_window_width, scaled_window_height))

# Function to find if a point is collision-free
def is_collision_free(new_node):
    for square in green_squares:
        if square[0] - 60 <= new_node[0] <= square[0] + 60 and square[1] - 60 <= new_node[1] <= square[1] + 60:
            return False
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
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(tree[start_index], tree[goal_index]), start_index))

    came_from = {}
    g_score = {node_index: np.inf for node_index in range(len(tree))}
    g_score[start_index] = 0
    f_score = {node_index: np.inf for node_index in range(len(tree))}
    f_score[start_index] = heuristic(tree[start_index], tree[goal_index])

    while open_set:
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


# Build RRT
tree, parent_map = build_rrt(start, goal, num_iterations=1000, step_size=50)

# A* search for the shortest path
goal_index = np.argmin(np.linalg.norm(tree - red_cylinder, axis=1))
path_indices = a_star_search(tree, 0, goal_index, parent_map)
if path_indices:
    a_star_path = tree[path_indices]
    print("A* Shortest Path Coordinates:")
    for node in a_star_path:
        print(node)
        # splprep prepares the data for smoothing by constructing a B-spline representation
    tck, u = interpolate.splprep([a_star_path[:, 0], a_star_path[:, 1]], s=0)
    # splev performs the smoothing operation by evaluating the B-spline at various points to create a smoother path for the car to navigate
    smoothed_path = interpolate.splev(np.linspace(0, 1, 100), tck)

# Draw the environment and A* path
draw_environment()

# Draw the smoothed path
if path_indices and smoothed_path is not None:
    for i in range(1, len(smoothed_path[0])):
        start_pos = (int(smoothed_path[0][i-1]), int(smoothed_path[1][i-1]))
        end_pos = (int(smoothed_path[0][i]), int(smoothed_path[1][i]))
        pygame.draw.line(window, (255, 0, 0), start_pos, end_pos, 5)


# Variables to track car position and rotation
car_x, car_y = start  # Initial car position
rotation_angle = 0  # Initial rotation
current_target_index=0
# Main loop
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the window
    window.fill((255, 255, 255))

    # Draw the environment
    draw_environment()
    # Draw the smoothed path
    if path_indices:
        for i in range(1, len(smoothed_path[0])):
            #  Retrieves the x and y coordinates of the starting point of the line segment. 
            start_pos = (int(smoothed_path[0][i-1]), int(smoothed_path[1][i-1]))
            # Retrieves the x and y coordinates of the ending point of the line segment. 
            end_pos = (int(smoothed_path[0][i]), int(smoothed_path[1][i]))
            pygame.draw.line(window, (255, 0, 0), start_pos, end_pos, 10)

     # Move the car along the smoothed path
    if path_indices and current_target_index < len(smoothed_path[0]):
        next_point = (smoothed_path[0][current_target_index], smoothed_path[1][current_target_index])
        # These differences represent the displacement vector from the current position to the next point,
        # which is crucial for determining the direction and distance the car needs to move in order to reach the next point.
        dx, dy = next_point[0] - car_x, next_point[1] - car_y
        # np.arctan2(dy, dx) to compute the angle between the line connecting the current position and the next point and the x-axis. 
        # np.degrees() converts the angle from radians to degrees, and - 90 is subtracted to adjust the angle so that the car's forward direction is oriented correctly.
        rotation_angle = np.degrees(np.arctan2(dy, dx)) - 90
        distance_to_next_point = np.sqrt(dx**2 + dy**2)

        if distance_to_next_point > 3:  # If not close to the next point, move towards it
            speed = 5  # You can adjust this speed
            movement_vector = np.array([dx, dy]) / distance_to_next_point * speed
            car_x += movement_vector[0]
            car_y += movement_vector[1]
        else:
            current_target_index += 1  # Move to the next poin

    # Draw the car
    rotated_car_img = pygame.transform.rotate(car_img, -rotation_angle)
    rotated_rect = rotated_car_img.get_rect(center=(car_x, car_y))
    window.blit(rotated_car_img, rotated_rect.topleft)

    # Update display
    pygame.display.flip()

    # Control frame rate
    clock.tick(30)  # Limit to 60 frames per second
