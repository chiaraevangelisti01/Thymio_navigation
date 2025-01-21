import numpy as np
import math as mt
import cv2

class Vision:
    """
    This class defines how the camera is to be used

    Attributes:
        _camID: the ID of the camera for opencv (could be string if a feed from an IP camera)
        _thymio_tagID: the aruco ID of the tag on the robot
        _calf_file: string to for storing the HSV calibration of the obstacle detection
        _areaPX: size of the camera feed and window
        _areaCM: size of the playground
        _hsv_buffer_size: size of the calibration buffer
        _obstacles_hsv: hsv bounds of the obstacles
        _detector: aruco tag detector
        _frame: current image
        _cap: opencv video capture object
        obstacles (list): List of detected obstacles.
        thymio_pos (list): Position of the robot.
        target (np.array): Target position for the robot.
        direction (np.array): Direction vector of the robot.
        usable_thymio_pos (bool): Flag indicating if the robot position is usable.
        usable_obstacles (bool): Flag indicating if the obstacle data is usable.

    """
    def __init__(self, camID, tagID, areaPX=(1080, 720), areaCM=(135, 90), calibration_filename="cal.npy", obs_min_area=250):
        """
        Initializes the Vision object with camera settings, area dimensions, and calibration data.

        Args:
            camID (int or string): ID of the camera to be used or camera IP link.
            tagID (int): ID of the ArUco tag associated with the robot.
            areaPX (tuple): Dimensions of the area in pixels (width, height).
            areaCM (tuple): Dimensions of the area in centimeters (width, height).
            calibration_filename (str): File name for saving and loading calibration data.
            obs_min_area (int): Minimum area in pixels to consider an object as an obstacle.
        """

        # Saving arguments
        self._obs_min_area=obs_min_area
        self._camID=camID
        self._thymio_tagID=tagID
        self._cal_file=calibration_filename


        self._cap=cv2.VideoCapture(camID)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self._areaPX=areaPX
        self._areaCM=areaCM

        cv2.namedWindow('Video Feed', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Video Feed',self._get_hsv_value)

        
        # Load the predefined dictionary of ArUco markers
        arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        arucoParams = cv2.aruco.DetectorParameters()

        # Create the detector object
        self._detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

        self._frame=None

        #creates the a buffer and tries to load already calibrated values
        self._hsv_buffer_size=30
        self._hsv_buffer = np.full((self._hsv_buffer_size, 3), np.nan)
        try:
            self._obstacles_hsv = np.load(self._cal_file)
        except FileNotFoundError:
            self._obstacles_hsv = None

        self.obstacles=None
        self.thymio_pos=None
        self.target=np.array([0,0])
        self.direction=None
        self.usable_thymio_pos=False
        self.usable_obstacles=False
        
    def get_thymio_pos(self):
        """
        Returns:
            list: The position of the robot in the format [x, y, angle] [cm] or None if the position is not usable.
        """
        if  not self.usable_thymio_pos: return None
        pos = self.thymio_pos
        factor=self._areaCM[0]/self._areaPX[0]
        pos = [pos[0]*factor,pos[1]*factor,pos[2]]
        return pos
    
    def get_obstacles(self):
        """
        Returns:
            list: List of transformed polygons representing obstacles [cm] or None if obstacles data is not usable.
        """
        # Check that we have already seen obstacles
        if not self.usable_obstacles: return None
        # Corrects for the real size in cm and returns the polygons pos in cm
        factor=self._areaCM[0]/self._areaPX[0]
        transformed_polygons = []
        for polygon in self.obstacles:
            transformed_polygon = [([i[0][0] * factor,i[0][1] * (factor)]) for i in polygon]
            transformed_polygons.append(transformed_polygon)
        return transformed_polygons
    
    def get_target(self):
        """
        Returns:
            list: The target position in the format [x, y].
        """
        pos = self.target
        # Corrects for the real size in cm
        factor=self._areaCM[0]/self._areaPX[0]
        return [pos[0]*factor,pos[1]*factor]
    
    def update(self):
        """
        Updates the vision system by capturing a new frame and processing it.
        Detects the position of the robot, obstacles, and the target. This needs to be called at each loop
        """

        # Init usable as False in each loop
        self.usable_thymio_pos=False
        self.usable_obstacles=False

        # Gets the frame and crashed if no image
        ret, self._frame = self._cap.read()
        if not ret:
            raise ValueError("Camera Missing")

        # Tries to warp the image on the corners, then seeks for target and thymio and tries to draw them
        if self._warp():
            self.usable_obstacles=self._update_obstacles()
            self.usable_thymio_pos=self._update_thymio()
            cv2.circle(self._frame, (tuple(self.target[:2])), 5, [0,0,255],-1)
            if self.usable_obstacles: self._draw_contours()
            if self.usable_thymio_pos: self._draw_robot()

        # Show image
        cv2.imshow('Video Feed', self._frame)

        # Necessary for opencv to work
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pass


    def _get_mask(self, image, colors):
        #This return thresholded image with 2 colors
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, colors[0], colors[1])
        return mask

    def _warp(self):

        # Getting aruco ids and corners
        (corners, ids, rejected) = self._detector.detectMarkers(self._frame)

        # Check if there are four markers detected

        centers=[0,0,0,0]

        # Sort the corners if necessary. The order should be: top-left, top-right, bottom-right, bottom-left
        id_seen=[]
        if ids is None: return False

        for corner, id in zip(corners, ids):
            # Getting corners
            if id[0] in [0,6,2,3]:
                center = np.mean(corner[0], axis=0)
                val=id[0]
                if val==6: val=1
                centers[val] = center
                id_seen.append(id)
            #getting calibration marker
            if id == 5:
                center = np.mean(corner[0], axis=0)
                pos = center+ corner[0,1]-corner[0,2]
                self._update_hsv_obs(pos)
        # check is all the corners are seen
        if sorted(id_seen) != [0,2,3,6]:
            return False
        # convert to float32
        pts1 = np.float32([c for c in centers])
        # Define the points for the transformed image
        _width=self._areaPX[0]
        _height=self._areaPX[1]
        pts2 = np.float32([[0, 0], [_width, 0], [_width, _height], [0, _height]])

        # Compute the perspective transform matrix and apply it
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        self._frame = cv2.warpPerspective(self._frame, matrix, (_width, _height))

        return True
    
    def _update_obstacles(self):
        self.obstacles=[]
        # If we don't have HSV bounds we don't know what to look for
        if self._obstacles_hsv is None:
            return False
        
        # Get the thresholded image and find the contours
        obs_mask = self._get_mask(self._frame, self._obstacles_hsv)
        temp_contours, dump = cv2.findContours(obs_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Get rid of the too smal obstacles
        obs=[]
        for i in temp_contours:
            if cv2.contourArea(i)> self._obs_min_area:
                obs.append(i)


        # Limit the number of corners of the polygons for djikstra
        epsilon_factor = 0.02  # This factor controls the approximation precision
        for cnt in obs:
            epsilon = epsilon_factor * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) > 2:  # Filter out small shapes, adjust as needed
                self.obstacles.append(approx)
        return True

    def _update_thymio(self):
        # Get arucos make sure to have something
        (corners, ids, rejected) = self._detector.detectMarkers(self._frame)        
        center=np.array([0,0])
        time_seen=0
        direction=None
        if ids is None: return

        # Detect thymio pos and target, checks if tymio was seen
        for corner, id in zip(corners, ids):
            if id==self._thymio_tagID:
                center = np.mean(corner[0], axis=0)
                time_seen+=1
                direction=corner[0][0]-corner[0][1]
            if id == 7:
                self.target=np.mean(corner[0], axis=0).astype(np.int64)
        if time_seen != 1: return False

        # Update pos and dir
        self.direction=np.array([-direction[1], direction[0]])
        angle=mt.atan2(*self.direction[::-1])
        self.thymio_pos=[int(center[0]), int(center[1]), angle]


        return True

    def _draw_contours(self):
        for cnt in self.obstacles:
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            cv2.drawContours(self._frame, [approx], -1, (0, 255, 0), 3)

    def _draw_robot(self):
        cv2.circle(self._frame, (tuple(self.thymio_pos[:2])), 5, [0,0,255],-1)
        start=tuple(self.thymio_pos[:2])
        stop =tuple((self.thymio_pos[:2] + self.direction).astype(int))
        cv2.line(self._frame,start,stop,(255,0,0),5)

    def _get_hsv_value(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Get the HSV value at the clicked point
            hsv_frame = cv2.cvtColor(self._frame, cv2.COLOR_BGR2HSV)
            hsv_value = hsv_frame[y, x]
            print("HSV Value at ({}, {}): {}".format(x, y, hsv_value))
    
    def _update_hsv_obs(self, pos):
        # Function called when the calibration marker is seen
        
        # Converts image to hsv
        hsv_frame = cv2.cvtColor(self._frame, cv2.COLOR_BGR2HSV)
        pos=pos.astype(np.int64)

        # Checks for out of bounds
        if pos[0]>self._areaPX[0]: return
        if pos[1]>self._areaPX[1]: return

        # Draws the position the color is taken from
   
        # Gets HSV value from the point
        middle_point = hsv_frame[*pos[::-1]]
        # Draws the position the color is taken from
        cv2.circle(self._frame,pos, 5, [0,255,0],-1 )
        # Adds the value to a rolling buffer from which the mean is the derived
        self._hsv_buffer[0]=middle_point
        self._hsv_buffer=np.roll(self._hsv_buffer, 1, axis=0)
        hsv_bounds = np.array([[0, 0, 0], [180, 255, 220]])  # Assuming Hue is in 0-180
        mean=np.nanmean(self._hsv_buffer, axis=0 )
        # Margin tolerance
        margin=[10, 50, 50]

        # Calculate lower and upper bounds
        lower_bound = np.clip(mean - margin, hsv_bounds[0], hsv_bounds[1])
        upper_bound = np.clip(mean + margin, hsv_bounds[0], hsv_bounds[1])

        # HSV margin array
        self._obstacles_hsv = np.array([lower_bound, upper_bound])
        np.save(self._cal_file, self._obstacles_hsv)


if __name__ == '__main__': 
    vis=Vision(0, 4) #Camera id left and Thymio ID right
    while True:
        vis.update()
        #from there we could for exemple vis.get_target() or call other functions