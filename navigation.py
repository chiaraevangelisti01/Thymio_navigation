import numpy as np
import random
import matplotlib.pyplot as plt
import pyvisgraph as vg
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union, orient
import geopandas as gpd
import time

class Navigation:
    def __init__(self,map_size=(135, 90), k_att=1, k_rep=20, k_rep_unpredicted= 55, safety_radius=4.0, 
                 step_size=40, epsilon=1, threshold = 10):
        self.k_att = k_att                                        #constant modulative the attractive potential field                
        self.k_rep = k_rep                                        #constant regulating the repulsive potential field of fixed obstacles
        self.k_rep_unpredicted = k_rep_unpredicted                #constant regulating the repulsive potential field of moving obstacles
        self.safety_radius = safety_radius                        #safety constant to enlarge the obstacles
        self.robot_radius = 8                                     #approximate robot radius
        self.step_size = step_size                                #constant to regulate the impact of gradient 
        self.epsilon = epsilon                                    #step to calculate the finite difference
        self.delta_angle = 0.436                                  #approximate angle between sensors
        self.prox_sensor_threshold = threshold                    #threshold for sensor values
        self.detection_dist = 8                                   #dist at which the obstacle is detected [cm]
        self.last_local_nav_time_trigerred=-5                     #last time local nav vas trigered
        self.local_nav_time= 3                                    #minimum time to stay in local nav [s]

        self.obs_std=30
        self.prox_sensor_readings = None                          #values of the frontal horizontal proximity sensors 
        self.map_size = map_size                                  #size of the environment

        #Create the grid from the map
        self.x_range = np.linspace(0, self.map_size[0], self.map_size[0])
        self.y_range = np.linspace(0,self.map_size[1], self.map_size[1])
        self.X, self.Y = np.meshgrid(self.x_range, self.y_range)
      
        #Common ------------------------------------------------------------------------------------------------------------------------------
        self.polygons=None                                        #list of coordinates of the polygons, coming from vision
        self.ax=None

        #Global nav attributes ---------------------------------------------------------------------------------------------------------------
        self.env=None                                             #structure of the environment created
        self.dilated_env=None                                     #environment dilated
        self.vg_graph=None                                        #visiblity graph object
        self.path=None                                            #patj calculated

#-----------------------------------------------------INTERFACE METHODES--------------------------------------------------------------
    def update(self, prox_sensor_readings, polygons, current_pos, target):

        #Read the current sensor reading and decide if local navigation should be entered
        self.prox_sensor_readings = prox_sensor_readings
        local_nav= (sum(i> self.prox_sensor_threshold for i in self.prox_sensor_readings)>0)

        #Read the information from the camera and update the environment if new data is given
        updated_geom=False
        if polygons is not None:
            self.update_polygons(polygons)
            self.update_environment()
            self.update_dilated_env()
            updated_geom=True
        if self.polygons is None: return

        
        #Call local if sensors detected and the time in local nav has elapsed
        if( local_nav and not time.time()-self.last_local_nav_time_trigerred<self.local_nav_time):
            self.last_local_nav_time_trigerred=time.time()
            return self.local_navigation(current_pos, target)
        
        #Call local if the time in local nav has not elapsed yet
        if( time.time()-self.last_local_nav_time_trigerred<self.local_nav_time ):
            return self.local_navigation(current_pos, None)
        
        #Call global
        else:
            return self.global_navigation(current_pos, target, updated_geom)
    
    def local_navigation(self, current_pos, target):
        #Return an angle to be followed for the PD controller 

        #Update the field only if the local nav time has elapsed
        if target is not None:
            self.update_potential_field(current_pos, target)

        #Find the direction of motion
        next_point = self.find_next_position(current_pos)
        self.path = [next_point]                             
        self.plot_geoenv(current_pos)

        return self.path
        
    def update_potential_field(self, current_pos, target):
        #Update the potential field thorugh the necessary steps
        potential_map = self.potential_field_map(self.env, target)
        repulsive_potential = self.potential_field_temp(current_pos)
        self.potential_field = self.calculate_total_potential(potential_map, repulsive_potential)

    def global_navigation(self, current_pos, target , updated):
        #Return a path based on the visibility graph from the detected environment
        expanded_polygons = self.extract_coordinates()
        
        if not any(geometry.contains(Point(current_pos)) for geometry in self.dilated_env):
            if updated: self.create_vg_graph(expanded_polygons)
            vs_path = self.shortest_path(current_pos, target)
            path_gs, self.path = self.convert_path_vs(vs_path)
            self.plot_geoenv(current_pos)

        return self.path


#-----------------------------------------------------LOCAL--------------------------------------------------------------
    def potential_field_map(self, geo_env, target):
        #Creates the potential map as a matrix on the base of the environment given
        potential_map = np.zeros_like(self.X)

        #Iterate over all the dicretized points of the map
        for col in range(len(self.x_range)):
            for row in range(len(self.y_range)):
                current_point = Point(self.x_range[col], self.y_range[row])
                
                #Attractive contribution due to the goal
                att_force = self.k_att * np.linalg.norm([target[0] - current_point.x, target[1] - current_point.y])

                #Repulsive contribution due to the fixed obstacles
                rep_force=0
                for obstacle in geo_env:
                    distance = current_point.distance(obstacle)
                    if distance < self.robot_radius + self.safety_radius:
                        if int(distance) != 0:
                            rep_force += self.k_rep * np.exp(-0.5 * ((distance+self.obs_std/3) / (self.obs_std)) ** 2)
                        else:
                            rep_force +=self.k_rep
                
                #Total potential contribution
                potential_map[row, col] = att_force + rep_force

        return potential_map

    def calculate_single_repulsive_potential(self, obstacle_position):
        #Calculate the repulsive field due to local obstacle (Gaussian)
        distance = np.sqrt((self.X - obstacle_position[0])**2 + (self.Y - obstacle_position[1])**2)
        return  self.k_rep_unpredicted * np.exp(-0.5 * ((distance+self.obs_std/3) / self.obs_std) ** 2)

    def potential_field_temp(self, current_pos):
        #Calculate the potential map as a matrix on the base of the proximity sensor readings

        current_angle=current_pos[2]
        rep_dist =  self.detection_dist + self.robot_radius 

        #Find the angles at which the obstacles are detected and take the mean
        angles=[]
        for i in range(-2, 3):
            if self.prox_sensor_readings[i+2]>self.prox_sensor_threshold:
                angles.append(current_angle - i*self.delta_angle)
        angle=np.mean(angles)

        pos=[current_pos[0]+np.cos(angle)* rep_dist, current_pos[1] + np.sin(angle)*rep_dist]

        return self.calculate_single_repulsive_potential(pos)

    def calculate_total_potential(self, potential_map, repulsive_potential):
        #Sum the temporary potential local map and the gloal one
        total_potential = potential_map + repulsive_potential
        return total_potential

    def compute_gradient(self, current_X_pos, current_Y_pos, total_potential_map):
        #Calculate the gradients as finite differences
        df_dx = (total_potential_map[round(current_Y_pos), round(current_X_pos + self.epsilon)] - total_potential_map[round(current_Y_pos), round(current_X_pos - self.epsilon)]) 
        df_dy = -(total_potential_map[round(current_Y_pos + self.epsilon), round(current_X_pos)] - total_potential_map[round(current_Y_pos - self.epsilon), round(current_X_pos)])
        return np.array([df_dx, df_dy])

    def find_next_position(self, current_pos):
        #Return the angle towaards the next position
        gradient = self.compute_gradient(current_pos[0], current_pos[1],self.potential_field) 
        direction = -(gradient / np.linalg.norm(gradient))
        return np.arctan2(direction[1], direction[0])

    def plot_potential_map_arrows(self, current):
        #Function to plot the potential
        rows, cols = self.potential_field.shape
        x_range = np.linspace(0, cols - 1, cols)
        y_range = np.linspace(0, rows - 1, rows)
        X, Y = np.meshgrid(x_range, y_range)

        gradient_x = np.gradient(self.potential_field, axis=1)
        gradient_y = np.gradient(self.potential_field, axis=0)
        arrow_magnitudes = np.sqrt(gradient_x**2 + gradient_y**2)
        arrow_directions_x = gradient_x / arrow_magnitudes
        arrow_directions_y = gradient_y / arrow_magnitudes

        plt.figure(figsize=(20, 10))
        plt.quiver(X, Y, -arrow_directions_x, arrow_directions_y, scale=5, color='r', scale_units='inches', width=0.001)
        plt.scatter(current[0], current[1], c='b', marker='o', label='Current')
        #plt.scatter(target[0], target[1], c='g', marker='o', label='Goal')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.gca().invert_yaxis()
        plt.legend()
        plt.show()
        plt.pause(10000)


#-----------------------------------------------------GLOBAL--------------------------------------------------------------
    def update_polygons(self, list_coordinates):
        #Create polygons objects using shapely, from a list of the coordinates of the corners coming from vision
        self.coo=list_coordinates

        polygons = []
        for cor in list_coordinates:
            polygons.append(Polygon(cor))

        self.polygons=polygons

    def update_environment(self):
        #Create the environment from the polygons as a GeoSeries structure
        self.env = gpd.GeoSeries(self.polygons)

    def update_dilated_env(self):
        #Dilate the environment
        temp = self.env.buffer(resolution=1, distance=self.safety_radius+ self.robot_radius)

        #Create the walls as an obstacle
        polygon_coords = [(0, 0), (0, self.map_size[1]), (self.map_size[0], self.map_size[1]), (self.map_size[0], 0)]
        polygon = Polygon(polygon_coords)
        box = polygon.buffer(20,join_style=2)
        external_area = box.difference(polygon)

        # Create a rectangle for the opening
        opening = Polygon([[1,1],[1,-50], [-50,-50], [-50,1]])

        # Subtract the opening from the external area
        external_area = external_area.difference(opening)

        #Add the walls to the environment with the dilatd polygons
        self.dilated_env= gpd.GeoSeries([unary_union([*temp, external_area])])


    def extract_coordinates(self):
        #Extract the coordinates from a Geoseries environment

        list_coordinates = []

        for geom in self.dilated_env:
                # Check if the geometry is a Polygon
            if geom.geom_type == 'Polygon':
                x, y = geom.exterior.coords.xy
                list_coordinates.append(list(zip(x, y)))

            # Check if the geometry is a MultiPolygon
            elif geom.geom_type == 'MultiPolygon':
                for polygon in geom.geoms:
                    x, y = polygon.exterior.coords.xy
                    list_coordinates.append(list(zip(x, y)))
            
        return list_coordinates

    def create_vg_graph(self, list_coordinates):
        #Create the visibility graph from the expanded environment

        vg_obstacles = []
        for set in list_coordinates:
            
            pol = []
            
            for el in set:
                ex = random.uniform(-0.01, 0.01)  # NOTE:needs to be added because pyvisgraph vision has issues when a segment is the prolongation of another one
                ey = random.uniform(-0.01, 0.01)
                pol.append(vg.Point(el[0]+ex,el[1]+ey))
            
            vg_obstacles.append(pol)
        
        self.vg_graph = vg.VisGraph()
        self.vg_graph.build(vg_obstacles, workers=1, status=False)

    def shortest_path(self, start, goal):
        #Find the shortest path from the current position to the goal, based on the visibility graph

        start_vg = vg.Point(start[0],start[1])
        goal_vg = vg.Point(goal[0],goal[1])
        
        vs_path = self.vg_graph.shortest_path(start_vg, goal_vg)

        return vs_path

    def convert_path_vs(self, vs_path):
        #Convert the path from vs format to shapely and tuples
        path_gs = []
        path_tuples = []

        for el in vs_path:

            path_gs.append(Point(el.x,el.y))
            path_tuples.append((el.x,el.y))
        
        return path_gs, path_tuples


    def plot_geoenv(self, current_pos):
        #Plot the environment with the path

        if self.ax is None:
            fig, self.ax = plt.subplots()  
        self.ax.clear()

        
        self.dilated_env.plot(ax=self.ax)  # Plot on the created axes
        if not self.env.empty: self.env.plot(ax=self.ax, color=[1, 0, 0, 1])  
        if not (len(self.path)==1):
            x_coords, y_coords = zip(*self.path)
            self.ax.plot(x_coords, y_coords, linestyle='-', color='blue')  

        else:
            target= [np.cos(self.path[0])*20, np.sin(self.path[0])*20,]
            
            dir_angle=current_pos[2]
            direction=[np.cos(dir_angle)*20, np.sin(dir_angle)*20]
            self.ax.arrow(*current_pos[0:2], *target, color='red', width=1)
            self.ax.arrow(*current_pos[0:2], *direction, color='blue', width=1)

        plt.ion()
        plt.xlim(0, self.map_size[0]) 
        plt.ylim(0, self.map_size[1])
        self.ax.invert_yaxis()
        plt.show()  
        plt.pause(0.005)

    
#Space for trying the code 
if __name__=="__main__":
    nav= Navigation()
    O1 = [(40,20),(50,40),(110, 30)]
    O2 = [(180,200),(150,150),(200,40)]

    obstacles = [O1]
    i=0
    while(True):
        i+=3
        print(nav.update([0,0,1501,0,0], obstacles, (120,60,-np.pi/2), (20,20)))