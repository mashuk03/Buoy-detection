import numpy as np
import math

'''
Borrowed from https://github.com/AtsushiSakai/PythonRobotics/blob/master/Localization/extended_kalman_filter/extended_kalman_filter.py
'''

DT = 0.1

# Covariance for EKF simulation
Q = np.diag([
    0.90,  # variance of location on x-axis
    0.90,  # variance of location on y-axis
    np.deg2rad(0.90),  # variance of yaw angle
    0.90  # variance of velocity
]) ** 2 # predict state covariance
R = np.diag([0.20, 0.20]) ** 2 # Observation x,y position covariance


def motion_model(x, u):
    F = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 0]])

    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT],
                  [1.0, 0.0]])

    x = F @ x + B @ u

    return x


def observation_model(x):
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    z = H @ x

    return z


def jacob_f(x, u):
    """
    Jacobian of Motion Model
    motion model
    x_{t+1} = x_t+v*dt*cos(yaw)
    y_{t+1} = y_t+v*dt*sin(yaw)
    yaw_{t+1} = yaw_t+omega*dt
    v_{t+1} = v{t}
    so
    dx/dyaw = -v*dt*sin(yaw)
    dx/dv = dt*cos(yaw)
    dy/dyaw = v*dt*cos(yaw)
    dy/dv = dt*sin(yaw)
    """
    yaw = x[2, 0]
    v = u[0, 0]
    jF = np.array([
        [1.0, 0.0, -DT * v * math.sin(yaw), DT * math.cos(yaw)],
        [0.0, 1.0, DT * v * math.cos(yaw), DT * math.sin(yaw)],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]])

    return jF


def jacob_h():
    # Jacobian of Observation Model
    jH = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    return jH


def ekf_estimation(xEst, PEst, z, u):
    #  Predict
    xPred = motion_model(xEst, u)
    jF = jacob_f(xEst, u)
    PPred = jF @ PEst @ jF.T + Q

    #  Update
    jH = jacob_h()
    zPred = observation_model(xPred)
    y = z - zPred
    S = jH @ PPred @ jH.T + R
    K = PPred @ jH.T @ np.linalg.inv(S)
    xEst = xPred + K @ y
    PEst = (np.eye(len(xEst)) - K @ jH) @ PPred
    return xEst, PEst

class EKFTracker:
    def __init__(self):
        self.xEst = np.zeros((4, 1))
        self.PEst = np.eye(4)
        self.ud = np.ones((2, 1))

    def set(self, pos):
        self.xEst[0] = pos[0]
        self.xEst[1] = pos[1]

    def update(self, raw):

        z = np.reshape(raw, (2, 1))
        xEst, self.PEst = ekf_estimation(self.xEst, self.PEst, z, self.ud)
        delta = xEst - self.xEst
        self.ud[0, 0] = delta[0, 0] / DT
        self.ud[1, 0] = delta[1, 0] / DT
        self.xEst = xEst

    def get(self):
        # print(self.xEst.shape)
        return observation_model(self.xEst).astype('int')
