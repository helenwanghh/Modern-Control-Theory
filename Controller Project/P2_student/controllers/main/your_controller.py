# Fill in the respective functions to implement the controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *

# CustomController class (inherits from BaseController)
class CustomController(BaseController):

    def __init__(self, trajectory):

        super().__init__(trajectory)

        # Define constants
        # These can be ignored in P1
        self.lr = 1.39
        self.lf = 1.55
        self.Ca = 20000
        self.Iz = 25854
        self.m = 1888.6
        self.g = 9.81
        self.error_integral = 0
        # Add additional member variables according to your need here.
        self.cumulative_error_distance = 0
        self.previous_error_distance = 0
         
        # Constants 
        self.speed_prefer = 20.0  # Preferred speed
        self.Kp_speed = 300.0
        self.Kd_speed = 5
        self.KI_speed = 0.03
        # self.Kd_angle = 0.001
        # self.KI_angle = 1.3

    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g
        
         # ---------------|Longitudinal Controller|-------------------------
        
        # ---------------| write PID for velocity| ---------------------
        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)

        #velocity = xdot
        velocity = np.sqrt(xdot**2 + ydot**2)*3.6  #km/h
        speed_error = self.speed_prefer - velocity
        
        self.previous_speed_error = speed_error 
        error_derivative = (speed_error - self.previous_speed_error)/delT #D 
        self.error_integral += speed_error

        # Fetch the states from the BaseController method

        # Design your controllers in the spaces below.
        # Remember, your controllers will need to use the states
        # to calculate control inputs (F, delta)

        forwardindex= 50  # makes the vehicle moving forward
        distance, closest_index = closestNode(X,Y,trajectory)
        if forwardindex + closest_index >= 8203:
           forwardindex=0

        X_desired = trajectory[closest_index + forwardindex,0]
        Y_desired = trajectory[closest_index + forwardindex,1]
        
        psi_desired = np.arctan2(Y_desired - Y, X_desired - X)
        # x_velocity=30 #desired velocity
                    
         #---------------|Lateral Controller|-------------------------
        
        
        
        #Please design your lateral controller below.
        
         #---------------|Pole Statement|-------------------------
        
        A = np.array([[0, 1, 0, 0], [0, -4*Ca / (m * xdot), 4*Ca / m, -(2*Ca*(lf - lr))/(m*xdot)], [0, 0, 0, 1], [0, -(2*Ca*(lf - lr)) / (Iz * xdot), (2*Ca*(lf - lr)) / Iz, (-2*Ca*(np.power(lf, 2) + np.power(lr, 2))) / (Iz * xdot)]])
        B = np.array([[0], [2*Ca / m], [0], [(2 * Ca* lf) / Iz]])
        P = np.array([-1, -2, -3, -4])
   
        
        K = signal.place_poles(A, B, P)
        K=K.gain_matrix
        
        
        error_e1 = distance
        error_e2 = wrapToPi(psi - psi_desired)
        error_e1_dot = error_e1 - self.previous_error_distance
        error_e2_dot = psidot

        e = np.hstack((error_e1, error_e1_dot, error_e2, error_e2_dot))

        delta = -np.matmul(K,e)
        delta=float(delta)


        """

        # ---------------|Longitudinal Controller|-------------------------
        """
        #Please design your longitudinal controller below.
        
       
        F = (speed_error * self.Kp_speed + error_derivative/ delT * self.Kd_speed + self.error_integral*delT * self.KI_speed)  #P 

        # Return all states and calculated control inputs (F, delta)
        return X_desired, Y_desired, xdot, ydot, psi_desired, psidot, F, delta
        
    #getTrajectory(your_controller)
