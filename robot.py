import math
import numpy as np
import kalman
import vision
import controller
import time
import navigation
from tdmclient import ClientAsync , aw # ClientAsync.aw allows us to run asychronous functions sychronously

# Function used to format the comm with the robot
def motors(left, right):
    return {
        "motor.left.target": [left],
        "motor.right.target": [right],
    }

class Robot:
    """
    A class representing a robot with capabilities such as movement control, obstacle detection, and position tracking.

    Attributes:
        wheelbase (float): Distance between the robot's wheels [cm]
        node: Thymio robot node for communication.
        client: Thymio communication client.
        distances (list): List of intensity measurements from proximity sensors.
        kal (KalmanFilter): Instance of a Kalman Filter for position estimation.
        vis (Vision): Vision object for visual processing and tracking.
        nav (Navigation): Navigation object for path planning.
        pd (PDController): PD Controller object for proportional-derivative control.
    """
    
    def __init__(self, wheelbase = 9.3):
        """
        Initializes the Robot

        Args:
            wheelbase (float): Distance between the robot's wheels in centimeters.
        """
        # Constants
        self.node = None                                 # Address of particular Thymio we are controlling
        self.client = None                               # Address of Thymio comminication client software
        self.wheelbase = wheelbase                       # Distance between wheels in centimeters
        

        # Robot State Variables                 # Target Motor Speed (0-500 where 500 ~ 20cm/s)
        self.distances = [0]*5      
        self.start_robot()

        # Inits objects
        self.kal = kalman.KalmanFilter(self.wheelbase)
        self.vis = vision.Vision(1,4) # Change first argument to 0 if using desktop (it is the camera ID)
        self.nav = navigation.Navigation(self.vis._areaCM)                                       
        self.pd = controller.PDController()
 

    def start_robot(self):
        """
        Initiate communication with Thymio client and robot node.
        """
        self.client = ClientAsync()                 # Create client object
        self.client.process_waiting_messages()      # Ensure connection to TDM (Thymio Device Manager)
        self.node = aw(self.client.wait_for_node()) # Ensure connection with Thymio robot (node)
        aw(self.node.lock())                       # Lock robot to change variables and run programs

    def update(self): # There is no use of kalman in a correct way. Eventually the robot's state varaibles are in the kalman object.
        """
        Main function to handle the update loop
        """
        self.get_motor_speed()
        self.update_prox_sensors()
        self.vis.update()
        # Feed Kalman in order to update kalman's position
        self.kal.update(self.vis.get_thymio_pos(),self.current_measured_speed)

        # Feed navigation to have a path, the path can be a list of points OR list with just an float (angle) in it.
        path=self.nav.update(self.distances, self.vis.get_obstacles(), self.kal.get_state(), self.vis.get_target(),)

        # PD controller (P IRL)
        speeds=self.pd.update(self.kal.get_state(), path)
        self.set_motor_speed(*speeds)



    def turn_off_robot(self):
        """
        Run when turning off robot.
        """
        aw(self.node.unlock())

    ### HIGH LEVEL FUNCTIONS ###


    def update_prox_sensors(self):
        """
        Description:
        Reads front proximity sensors and returns calculated distances in 

        Params:

        Returns:
        Raw sensor values as array [0, 1, 2, 3, 4] from left to right. (See Thymio Cheat Sheet for diagram)
        """
        prox = self.get_prox_sensors() # This is a special type and doesnt want to become a number. need to fix
        #distances = np.array([0,0,0,0,0])
        numSensors = 5
        for i in range(numSensors):
            self.distances[i] = prox[i]

    ### MEDIUM LEVEL FUNCTIONS ###
    def get_prox_sensors(self):
        """
        Params:

        Returns:
        Raw sensor values as array [0, 1, 2, 3, 4] from left to right. (See Thymio Cheat Sheet for diagram)
        """
        node = self.node
        client = self.client
        aw(node.wait_for_variables({"prox.horizontal"}))
        aw(client.sleep(0.01))
        prox = node.v.prox.horizontal
        return prox 

    def set_motor_speed(self,left,right):
        """
        Description: 
        Sets the target speeds of the left and right wheels of the Thymio.
        
        Params: 
        left - Left wheel target speed [int: -500 <= x <= 500] (500 ~ 20 cm/s)
        right - Right wheel target speed [int: -500 <= x <= 500] (500 ~ 20 cm/s)
        
        Returns: 
        Left and Right wheel action.
        """
        left=int(left)
        right=int(right)
        self.node.send_set_variables(motors(left, right))

    def stop_motor(self):
        """
        Description: 
        Sets the target speeds of the left and right wheels of the Thymio to 0.
        
        Returns: 
        Left and Right wheel action.
        """
        node = self.node
        aw(node.set_variables(motors(0,0)))

    def get_motor_speed(self):
        """
        Description: 
        Gets the:
        current speeds of the left and right wheels of the Thymio.

        Returns: 
        speed - [int left_speed, int right_speed] between -500 and 500 (500 ~ 20 cm/s)
        """
        node = self.node
        aw(node.wait_for_variables({"motor.left.speed"}))
        aw(self.client.sleep(0.01))
        spdl = node.v.motor.left.speed #TODO add param for wheel change
        spdr = node.v.motor.right.speed #TODO add param for wheel change
        self.current_measured_speed=[spdl/25, spdr/25]
        

    ### LOW LEVEL FUNCTIONS

if __name__ == '__main__': 
    robot = Robot()
    try:
        while True:
            robot.update()
    except KeyboardInterrupt: 
        robot.stop_motor()
        robot.turn_off_robot()
        pass
        
