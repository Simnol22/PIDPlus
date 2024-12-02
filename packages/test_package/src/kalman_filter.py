import numpy as np
class KalmanFilter():
    def __init__(self, F, B, H, Q, R, x0, P0):
        self.F = F
        self.B = B  
        self.H = H  
        self.Q = Q  
        self.R = R  
        self.x = x0  
        self.P = P0  

    def predict(self, u_t):
        self.x = self.F @ self.x + self.B @ u_t
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x
        
    # To multiply a 2D matrix and a 1D vector, use 'np.dot()' and not the '*' operator
    def update(self, z):
        #Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update the estimate via z and covariance matrix
        y = z - self.H @ self.x
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = ((I - K @ self.H) @ self.P) @ (I - K @ self.H).T + K @ self.R @ K.T
        return self.x 
