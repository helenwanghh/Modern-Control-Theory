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
        self.speed_prefer = 20   #prefer speed
        self.error_integral = 0
        self.previous_speed_error = 0
        self.previous_angle_error = 0


        # Add additional member variables according to your need here.

    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g
       
        
        
      # Constants 
        self.speed_prefer = 20.0  # Preferred speed
        self.Kp_speed = 2000.0
        self.Kd_speed = 4.0
        self.KI_speed = 0.001
        self.Kp_angle = 1.2
        self.Kd_angle = 0.001
        self.KI_angle = 1.3



        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)
        
        #write PID for velocity 
        velocity = np.sqrt(xdot**2 + ydot**2)*3.6  #km/h
        speed_error = self.speed_prefer - velocity
        
        error_derivative = (speed_error - self.previous_speed_error)/delT  #D 
        self.error_integral += speed_error*delT
                
        # Design your controllers in the spaces below. 
        # Remember, your controllers will need to use the states
        # to calculate control inputs (F, delta). 

        # ---------------|Lateral Controller|-------------------------
        
	#set index
        min_distance, min_index = closestNode(X,Y,trajectory)
        index = min_index + 30
        if index>=  8203:
            index = 20

	#build trajectory 
        X_distance = trajectory[index][0] - X
        Y_distance = trajectory[index][1] - Y
        
        
     	#write PID for angle 
        theta = np.arctan2(Y_distance , X_distance)
        theta = wrapToPi(theta)
        
        angle_error = theta - psi
        angle_error = wrapToPi(angle_error)
        delta = angle_error
        

        # ---------------|Longitudinal Controller|-------------------------
        F = (speed_error * self.Kp_speed + error_derivative * self.Kd_speed + self.error_integral * self.KI_speed)  #P 
       
        self.previous_speed_error = speed_error 
        
        
        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta