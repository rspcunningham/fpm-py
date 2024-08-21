import numpy as np

def order_points_spirally(points):
    """
    Orders a sequence of points in 2D space according to a spiral pattern starting from (0,0).

    Parameters:
    points (numpy.ndarray): A 2D array of shape (num_points, 2) containing the points to order.

    Returns:
    numpy.ndarray: A 2D array of shape (num_points, 2) containing the points ordered in a spiral pattern.
    """
    # Calculate the distance of each point from the origin (0,0)
    distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    
    # Calculate the angle of each point relative to the origin
    angles = np.arctan2(points[:, 1], points[:, 0])
    
    # Combine distances and angles into a structured array for sorting
    structured_array = np.array(list(zip(distances, angles, points)), 
                                dtype=[('distance', float), ('angle', float), ('point', float, 2)])
    
    # Sort by distance first, then by angle
    sorted_structured_array = np.sort(structured_array, order=['distance', 'angle'])
    
    # Extract the ordered points
    ordered_points = np.array([item[2] for item in sorted_structured_array])
    
    return ordered_points

def generate_grid_points(num_points: int, max_radius: float) -> np.ndarray:
    """
    Generates a sequence of points in 2D space within a defined square, 
    ensuring that the outermost points lie on the boundaries.

    Parameters:
    min_x (float): Minimum x-coordinate of the square.
    max_x (float): Maximum x-coordinate of the square.
    min_y (float): Minimum y-coordinate of the square.
    max_y (float): Maximum y-coordinate of the square.
    num_points (int): Number of points to generate.

    Returns:
    numpy.ndarray: A 2D array of shape (num_points, 2) containing the generated points.
    """
    max_x, max_y = max_radius, max_radius
    min_x, min_y = -max_radius, -max_radius
    
    # Generate the x and y coordinates of the points
    x = np.linspace(min_x, max_x, int(np.sqrt(num_points)))
    y = np.linspace(min_y, max_y, int(np.sqrt(num_points)))
    
    # Create a meshgrid of the x and y coordinates
    X, Y = np.meshgrid(x, y)
    
    # Flatten the meshgrid to a 2D array of points
    points = np.column_stack((X.flatten(), Y.flatten()))
    
    return order_points_spirally(points)


def generate_fermat_spiral(num_points: int, max_radius: float) -> np.ndarray:

   # Golden angle in radians
    golden_angle = np.pi * (3 - np.sqrt(5))

    # Generate the points
    theta = np.arange(num_points) * golden_angle
    r = max_radius * np.sqrt(np.arange(num_points) / num_points)
    
    # Convert polar coordinates (r, theta) to Cartesian coordinates (x, y)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    return np.column_stack((x, y))

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_grid_points(points, max_radius):
    """
    Plots a set of points in 2D space and shades the overall box containing the points.

    Parameters:
    points (numpy.ndarray): A 2D array of shape (num_points, 2) containing the points to plot.
    min_x (float): Minimum x-coordinate of the box.
    max_x (float): Maximum x-coordinate of the box.
    min_y (float): Minimum y-coordinate of the box.
    max_y (float): Maximum y-coordinate of the box.
    """
    plt.figure(figsize=(8, 8))
    
    # Plot the points
    plt.scatter(points[:, 0], points[:, 1], c='blue', marker='o', label='Grid Points')
    
    # Annotate each point with its index
    for i, (x, y) in enumerate(points):
        plt.text(x, y, f'{i}', fontsize=9, ha='right', color='black')
    
    # Plot the shaded box
    rect = patches.Rectangle((-max_radius, -max_radius), max_radius*2, max_radius*2, 
                             linewidth=1, edgecolor='r', facecolor='lightgray', alpha=0.5)
    plt.gca().add_patch(rect)
    
    # Labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('2D Grid of Points with Bounding Box')
    plt.grid(True)
    plt.legend()
    
    # Show the plot
    plt.show()


# Example usage:
max_radius = 20
num_points = [10, 20, 50, 100]

led_positions = []

for n in num_points:
    points = generate_grid_points(n, max_radius)
    #plot_grid_points(points, max_radius)
    led_positions.append(points)

for n in num_points:
    points = generate_fermat_spiral(n, max_radius)
    #plot_grid_points(points, max_radius)
    led_positions.append(points)
