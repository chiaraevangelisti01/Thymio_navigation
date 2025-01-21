import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point

def potential_field_map(geo_env, goal_pos,map_size, k_att=0.1, k_rep=1.0, rep_radius=7.0):
    '''
    Inputs: environment created with Geoseries with the fixed obstacles, goal position, map size
            k_rep, k_att represents factor to give different weights to repulsive and attractive forces during computation
            rep_radius consider a minimum distance at which the repulsive field is activated'''
    
    #Create the empty grid
    x_range = np.linspace(0, map_size[0], map_size[0])
    y_range = np.linspace(0, map_size[1], map_size[1])
    X, Y = np.meshgrid(x_range, y_range)
    potential_map = np.zeros_like(X)

    #Fill the grid going over each point: attractive force everywhere based on the goal, repulsive force is close to an obstacle
    for i in range(len(x_range)):
        for j in range(len(y_range)):
            current_point = Point(x_range[i], y_range[j])
            att_force = k_att * np.array([goal_pos[0] - current_point.x, goal_pos[1] - current_point.y])

            rep_force = np.zeros(2)
            for obstacle in geo_env:
                distance = current_point.distance(obstacle)
                
                if distance < rep_radius:
                    if int(distance) != 0:
                        rep_force += k_rep * ((1 / distance - 1 / rep_radius) / distance**2) * np.array([current_point.x - obstacle.centroid.x, current_point.y - obstacle.centroid.y])

            total_force = att_force + rep_force
            potential_map[j, i] = np.linalg.norm(total_force)
    return potential_map


def plot_potential_map_arrows(potential_map, goal_pos):

    '''
    Input: potential map as a matrix of the same size of the environement, goal position  as a tuple of x, y coordinates
    Output: plot of the potential field as arrows
    '''
    # Get the shape of the potential map to determine the ranges
    rows, cols = potential_map.shape
    
    # Generate x and y ranges based on the shape of the potential map
    x_range = np.linspace(0, cols - 1, cols)
    y_range = np.linspace(0, rows - 1, rows)
    
    # Create a uniform grid for plotting
    X, Y = np.meshgrid(x_range, y_range)
    
    # Calculate arrow directions using central differences
    gradient_x = np.gradient(potential_map, axis=1)
    gradient_y = np.gradient(potential_map, axis=0)

    # Normalize arrow directions
    arrow_magnitudes = np.sqrt(gradient_x**2 + gradient_y**2)
    arrow_directions_x = gradient_x / arrow_magnitudes
    arrow_directions_y = gradient_y / arrow_magnitudes

    #Create figure
    plt.figure(figsize=(20, 10))
 
    # Plot arrows using quiver plot
    plt.quiver(X, Y, -arrow_directions_x, -arrow_directions_y, scale=5, color='r', scale_units='inches', width=0.005)

    # Plot goal position
    plt.scatter(goal_pos[0], goal_pos[1], c='g', marker='o', label='Goal')

    # Set labels and legend
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()

    
    # Show the plot
    plt.show()


def plot_potential_map_arrows_on_geoseries(potential_map, geo_env, goal_pos):

    '''
    Input: potential map as a matrix of the same size of the environement, goal position  as a tuple of x, y coordinates, geoseries environement
    Output: plot of the potential field as arrows superimposed to the geoseries environement built
    '''

    # Get the shape of the potential map to determine the ranges
    rows, cols = potential_map.shape
    
    # Generate x and y ranges based on the shape of the potential map
    x_range = np.linspace(0, cols - 1, cols)
    y_range = np.linspace(0, rows - 1, rows)
    
    # Create a uniform grid for plotting
    X, Y = np.meshgrid(x_range, y_range)
    
    # Calculate arrow directions using central differences
    gradient_x = np.gradient(potential_map, axis=1)
    gradient_y = np.gradient(potential_map, axis=0)

    # Normalize arrow directions
    arrow_magnitudes = np.sqrt(gradient_x**2 + gradient_y**2)
    arrow_directions_x = gradient_x / arrow_magnitudes
    arrow_directions_y = gradient_y / arrow_magnitudes

    # Plot potential field as contour plot
    plt.figure(figsize=(20, 10))
    
    # Plot GeoSeries environment
    geo_env.plot(facecolor='none', edgecolor='k')

    # Plot arrows using quiver plot
    plt.quiver(X, Y, -arrow_directions_x, -arrow_directions_y, scale=20, color='r', scale_units='inches', width=0.005)

    # Plot goal position
    plt.scatter(goal_pos[0], goal_pos[1], c='g', marker='o', label='Goal')

    # Set labels and legend
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()

    # Show the plot
    plt.show()


def calculate_single_repulsive_potential(X, Y, obstacle_position, k_rep, robot_radius):
    '''
    Inputs: X,Y of the grid which is being used, obstacle position as a tuple of x and y, k_rep constant defining the strength of the repulsive force, robot radius
    Outputs: return grid a repulsive potential created by that obstacle in the entire environement
    '''

    # Calculate the repulsive potential generated by a single obstacle, when sensed with the sensors
    repulsive_potential = k_rep / ((X - obstacle_position[0])**2 + (Y - obstacle_position[1])**2 + 1e-6)
    repulsive_potential *= np.exp(-0.5 * ((X - obstacle_position[0])**2 + (Y - obstacle_position[1])**2) / robot_radius**2)
    return repulsive_potential


def potential_field_temp(robot_position, prox_sensor_readings, robot_angle, robot_radius,delta_angle,map_size,tol,scale_factor,k_rep=1):
    '''
    Inputs: robot position as a tuple with x and y coordinates, sensor readings, robot angle, delta angle indicating the deviation of right of left sensors from the center in degrees,map size as a tuple
            of x and y dimensions, tolerance on the sensor values to trigger the repulsion, scale factor to translate sensor value into distance in the map, k_rep defining the intensity of the repuslive force
    Outputs: potential field generated by the sensed obstacle as a matrix
    '''
    x_range = np.linspace(0, map_size[0], map_size[0])
    y_range = np.linspace(0, map_size[1], map_size[1])
    X, Y = np.meshgrid(x_range, y_range)
    
    repulsive_potential = np.zeros_like(X)

    #I condiser multiple sensors altoghether becasue I though this would me more smoooth, but we can try it out and see what happens
    #Sensing obstacle to the left
    if sum(prox_sensor_readings[0:3])> 3*tol:
        # Calculate the position of the obstacle in polar coordinates
        obstacle_distance = sum(prox_sensor_readings[0:2])/scale_factor #we need to find this factor experimentally
        obstacle_angle = np.radians(robot_angle + delta_angle)
    
    #Sensing obstacle to the right
    elif sum(prox_sensor_readings[1:4])> 3*tol:
        # Calculate the position of the obstacle in polar coordinates
        obstacle_distance = sum(prox_sensor_readings[1:3])/scale_factor #we need to find this factor experimentally
        obstacle_angle = np.radians(robot_angle)

    #Sensing obstacle to the front
    elif sum(prox_sensor_readings[3:6])> 3*tol:
        # Calculate the position of the obstacle in polar coordinates
        obstacle_distance = sum(prox_sensor_readings[3:5])/scale_factor #we need to find this factor experimentally
        obstacle_angle = np.radians(robot_angle - delta_angle)
    
    else:
        obstacle_distance = 0

    if obstacle_distance!=0:
        #Calculate the obstacle position in map coordinates 
        obstacle_x = robot_position[0] + obstacle_distance * np.cos(obstacle_angle)
        obstacle_y = robot_position[1] + obstacle_distance * np.sin(obstacle_angle)

        if 0 <obstacle_x<map_size[0] and  0<obstacle_y< map_size[1]:

            # Calculate the repulsive potential field for the obstacle
            repulsive_potential += calculate_single_repulsive_potential(X, Y, [obstacle_x, obstacle_y], k_rep, robot_radius)

    return repulsive_potential

def compute_gradient(x, y, epsilon,overall_map):
    ''' 
    Inputs: x, y coordinates of the current robot position, epsilon to avoid numerical instability, overall potential map field (temporary+fixed) as a matrix
    Outputs: gradient of the field at that location
    '''
    #Compute the gradient along the two direction as finite sums method
    df_dx = (overall_map[x + epsilon, y] - overall_map[x - epsilon, y]) 
    df_dy = (overall_map[x, y + epsilon] - overall_map[x, y - epsilon]) 

    return np.array([df_dx, df_dy])

def find_next_position(robot_position,step_size,epsilon,overall_map):
    '''
    Inputs: robot position as a tuple with x and y coordinates, step_size to modulate the movement with respect to the gradient, epsilon '''

    # Compute gradient at the current position
    gradient = compute_gradient(robot_position[0], robot_position[1],epsilon,overall_map)

    # Normalize the gradient to get the direction
    direction = gradient / np.linalg.norm(gradient)

    # Choose a step size
    step_size = 0.5

    # Calculate the next point
    next_point = robot_position + step_size * direction

    return(next_point)

if __name__ == "__main__":
    O1 = [(3,8),(5,5),(8,7),(6,10)]
    O2 = [(20,5),(23,7),(28,7),(24,2)]
    O3 = [(18,9),(16,12),(21,15),(23,11)]
    O4 = [(32,12),(34,15),(40,14),(37,11)]
    obstacles = [O1,O2,O3,O4]

    #Create the polygons
    polygons = create_polygons(obstacles)

    #Create and plot geoseries environment
    environment = GeoSeries(polygons)
    GeoSeries.plot(environment)

    #Dilate the obstacles
    robot_size = 1
    environment_dil = environment.buffer(robot_size/2, join_style = 2)
    GeoSeries.plot(environment_dil, color ='green')
