import numpy as np
import time

class KalmanFilter:
    def __init__(self, wheel_base=9.5):
        # Wheel base of the robot
        self.L = wheel_base

        # Process noise covariance
        q = 0.1 ** 2
        self.Q = q * np.eye(3)

        # Measurement noise covariance
        self.R =np.diag([0.02**2,0.02**2, np.deg2rad(5)**2])

        # Initial state estimate
        self.x_hat = np.zeros(3)

        # Initial covariance estimate
        self.P = np.eye(3)

        # Last time update was called
        self.last_time = time.time()

    def predict(self, u):
        # Calculate the current time and dt
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        # Extract the state for readability
        x, y, theta = self.x_hat

        # Calculate the average speed
        v = (u[0] + u[1]) / 2

        # Update state
        x += dt * v * np.cos(theta)
        y += dt * v * np.sin(theta)
        theta += dt * (u[0] - u[1]) / self.L

        # Update state estimate
        self.x_hat = np.array([x, y, theta])

        # Jacobian of the motion model
        F = np.array([[1, 0, -dt * v * np.sin(theta)],
                      [0, 1, dt * v * np.cos(theta)],
                      [0, 0, 1]])

        # Predict covariance
        self.P = np.dot(np.dot(F, self.P), F.T) + self.Q

    def update(self, z, u):
        # If the measurement is None, skip the update step
        self.predict(u)
        if z is None:
            return

        # Measurement model
        H = np.eye(3)

        # Kalman gain
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(np.dot(np.dot(H, self.P), H.T) + self.R))

        # Update state estimate
        self.x_hat += np.dot(K, (z - self.x_hat))

        # Update covariance estimate
        self.P = self.P - np.dot(np.dot(K, H), self.P)

    def get_state(self):
        return self.x_hat