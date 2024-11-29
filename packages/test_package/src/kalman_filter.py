class KalmanFilter():
    def __init__(self):
        pass

    def predict(A, B, Q, mu_t, u_t, Sigma_t):
        predicted_mu = A @ mu_t + B @ u_t
        predicted_Sigma = A @ Sigma_t @ A.T + Q
        return predicted_mu, predicted_Sigma
        
    # To multiply a 2D matrix and a 1D vector, use 'np.dot()' and not the '*' operator
    def update(H, R, z, predicted_mu, predicted_Sigma):
        residual_mean = z - H @ predicted_mu
        residual_covariance = H @ predicted_Sigma @ H.T + R
        kalman_gain = predicted_Sigma @ H.T @ np.linalg.inv(residual_covariance)
        updated_mu = predicted_mu + kalman_gain @ residual_mean
        updated_Sigma = predicted_Sigma - kalman_gain @ H @ predicted_Sigma
        return updated_mu, updated_Sigma