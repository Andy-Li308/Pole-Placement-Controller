import numpy as np
import scipy.signal as signal

class Controller:
    #pole placement control 
    def __init__(self, m, k, b, desired_poles):
        self.m = m
        self.k = k
        self.b = b
        self.desired_poles = desired_poles

        #gain matrix K for state feedback control
        self.K = np.zeros((1, 2)) # initialize K as a 1x2 matrix for state feedback control

        #define the state matrices for the state space representation of the system
        self.A = np.array([[0, 1], [-k/m, -b/m]])
        self.B = np.array([[0], [1/m]])

    def place_poles(self): 
        #use scipy's place_poles function to compute the gain matrix K
        result = signal.place_poles(self.A, self.B, self.desired_poles)
        self.K = result.gain_matrix
        #show the eigenvalues of the original A matrix and the closed-loop system (A - BK)
        print("Desired poles:", self.desired_poles)
        print("original A matrix:\n", self.A)
        print("Original A matrix eigenvalues:", np.linalg.eigvals(self.A))
        print("Closed loop matrix (A - BK):\n", self.A - self.B @ self.K)
        print("Closed-loop system eigenvalues:", np.linalg.eigvals(self.A - self.B @ self.K))
        print("Gain matrix K:", self.K)

    def control(self, t, state):
        #compute the control input using state feedback control law u = -Kx (x is the state vector, K is the gain matrix)
        u = -self.K @ state
        return u.item()  # return as scalar value for compatibility with simulation input function
