
import numpy as np
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0
        self.pid_action = np.array([0.3, 0.0]) # initial pid action

    def compute(self, error, dt):
        # Proportional term
        proportional = self.kp * error

        # Integral term
        self.integral += error * dt
        integral = self.ki * self.integral

        # Derivative term
        derivative = self.kd * (error - self.previous_error) / dt
        self.previous_error = error

        # PID output
        return proportional + integral + derivative

    def get_action(self, dist, dt):
        error = -dist 
        pid_output = self.compute(error, dt)
        self.pid_action[1] = -pid_output # np.clip(pid_output, -1.0, 1.0)
        return self.pid_action