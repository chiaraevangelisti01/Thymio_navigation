import pyvisgraph as vg
from shapely.geometry import Polygon, Point, LineString
from scipy.interpolate import CubicSpline, make_interp_spline, make_smoothing_spline, splrep
import numpy as np


def create_polygons(list_coordinates):
    '''
    Input: array of tuples of obstackle corners in xy coordinate system: SORTED
    Output: array of obstacles as polygon data structure
    '''

    polygons = []

    for cor in list_coordinates:
        polygons.append(Polygon(cor))

    return polygons

def extract_coordinates(environment):
    ''' 
    Input: geoseries environement made of polygons
    Output: array of tuples of obstacle corners in xy coordinate system
    '''

    list_coordinates = []

    for pol in environment:
        x,y = pol.exterior.coords.xy
        
        list_coordinates.append(list(zip(x, y)))
    
    return list_coordinates

def convert_pol_to_vs(list_coordinates):
    '''
    Input: array of tuples of obstacle corners in xy coordinate system
    Output: array of vg.Points of obstacle corners 
    '''

    vg_obstacles = []

    for set in list_coordinates:
        
        pol = []
        
        for el in set:
            pol.append(vg.Point(el[0],el[1]))
        
        vg_obstacles.append(pol)

    return vg_obstacles

def convert_path_vs(vs_path):
    '''
    Input: array of points in vs format constituting the shortest path 
    Output  array of points in shapely structure, array of tuples of path steps in xy coordinate system  
    '''
    path_gs = []
    path_tuples = []

    for el in vs_path:

        path_gs.append(Point(el.x,el.y))
        path_tuples.append((el.x,el.y))
    
    return path_gs, path_tuples

def smoothen_path(path_tuples):
    '''
    Inputs: array of tuples path steps in xy coordinate system
    Outputs. array of tuples of the steps of the smoothened path in xy coordinate system'''

    # Generate a finer set of points for a smoother curve
    x_points = path_tuples[:,0]
    x_finer = np.linspace(x_points[0], x_points[-1], 30)  # Adjust the number of points as needed

    # Create an interpolated spline interpolation on the finer set of points
    spline_interp = make_interp_spline(path_tuples[:,0], path_tuples[:,1]) 

    # Obtain the corresponding y-coordinates on the smoothed curve
    y_finer_interp = spline_interp(x_finer)
    
    path = np.matrix.transpose(np.array([x_finer,y_finer_interp]))

    return path
