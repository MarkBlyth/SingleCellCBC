import numpy as np


class _PID:
    """Error checking is provided in advance by the main Controller class,
    so no need to perform any here."""

    def __init__(self, kp, ki, kd, B_matrix, C_matrix, target):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.B_matrix = np.array(B_matrix).reshape((1, -1))
        self.C_matrix = np.array(C_matrix)
        self.target = target

    def __call__(self, x, t):
        this_error = np.dot(self.C_matrix, x) - self.target(t)

        # If this is the first call to the controller, do nothing
        if not "last_time" in self.__dict__:
            self.last_time = t
            self.last_error = this_error
            self.integral_error = 0
            return 0 * self.B_matrix

        # Calculate PID components
        dt = t - self.last_time
        if dt < 0:
            raise ValueError("Time must increase monotonously")
        integral = self.integral_error + 0.5 * (self.last_error + this_error) * dt
        derivative = 0 if dt == 0 else (this_error - self.last_error) / dt
        proportional = this_error
        print(proportional, integral, derivative)

        # Find control action
        control_action = -1 * (
            self.kp * proportional + self.ki * integral + self.kd * derivative
        )

        # Save results
        self.last_time = t
        self.last_error = this_error
        self.integral_error = integral

        return control_action * self.B_matrix
