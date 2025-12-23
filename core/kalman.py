import numpy as np
from filterpy.kalman import KalmanFilter

def init_kalman():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    # 测量矩阵 (只能测量 x, y)
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    kf.R = np.eye(2) * 0.1
    kf.Q = np.eye(4) * 0.01
    kf.P *= 1000
    return kf

def init_kalman_centerline(num_points):
    kalman_filters = []
    for _ in range(num_points):
        kalman_filters.append(init_kalman())
    return kalman_filters