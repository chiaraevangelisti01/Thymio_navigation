import math as mt
import numpy as np

class PDController:
    def __init__(self, kp=50, kd=1, default_speed=100, max_speed=200):
        self.max_speed=max_speed
        self.kp = kp  # Proportional gain
        self.kd = kd  # Derivative gain
        self.default_speed = default_speed
        self.previous_error = 0
        self.is_target_behind=False
        self.follow_angle=False

    def update(self, current_position, path):
        if path is None: return (0,0)
        self.follow_angle=(len(path)==1)
        spd=self.default_speed
        if self.follow_angle:
            dist=0
            control_signal = self._pd_control(current_position, path[0])
            spd=50
        else:
            dist=np.linalg.norm(np.array(current_position[0:2])-np.array(path[-1]))
            if dist<5:
                return (0,0)
            control_signal = self._pd_control(current_position, path[1])

        



        if self.is_target_behind:
            spd = 0  # 
        return self._calculate_wheel_speeds(control_signal, spd)
    
    def _calculate_error(self, current_position, target_position):
        # Calculate the Euclidean distance as error
        if not self.follow_angle:
            current_position=np.array(current_position)
            target_position=np.array(target_position)
            vector=target_position-current_position[:2]
            target_angle=mt.atan2(*vector[::-1])
        else: 
            target_angle=target_position
            print("PD c t", current_position[2], target_angle)
        angle_difference = target_angle-current_position[2]
        angle_difference = (angle_difference + np.pi) % (2 * np.pi) - np.pi
        self.is_target_behind = np.abs(angle_difference) > np.pi / 2
        print(angle_difference)
        return angle_difference

    def _pd_control(self, current_position, target_position):
        error = self._calculate_error(current_position, target_position)
        derivative = error + self.previous_error
        control_signal = self.kp * error - self.kd * derivative
        self.previous_error = error
        print("control_signal", control_signal)
        return control_signal
    

    def _calculate_wheel_speeds(self, control_signal, spd):
        # Implement the logic to calculate wheel speeds based on your robot's design
        # This is a placeholder for your specific implementation
        left_wheel_speed = np.clip(spd + control_signal, -self.max_speed, self.max_speed)
        right_wheel_speed = np.clip(spd - control_signal, -self.max_speed, self.max_speed)
        return int(left_wheel_speed), int(right_wheel_speed)
    def bound_value(value, min_value, max_value):
        return max(min_value, min(value, max_value))



if __name__=="__main__":
    def motors(left, right):
        return {
            "motor.left.target": [left],
            "motor.right.target": [right],
        }
    import vision
    vis = vision.Vision(0,4)
    pd  = PDController()
    from tdmclient import ClientAsync, aw # ClientAsync.aw allows us to run asychronous functions sychronously
    while True:
        vis.update()
        if vis.get_thymio_pos() is not None:
            break
    client = ClientAsync()                 # Create client object
    client.process_waiting_messages()      # Ensure connection to TDM (Thymio Device Manager)
    node = aw(client.wait_for_node()) # Ensure connection with Thymio robot (node)
    aw(node.lock())     
    while True:
        vis.update()
        pos=vis.get_thymio_pos()
        if pos is None: continue
        speeds=pd.update(pos, vis.get_target())
        node.send_set_variables(motors(*speeds))


            
