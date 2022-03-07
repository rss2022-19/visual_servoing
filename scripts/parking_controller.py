#!/usr/bin/env python

import rospy
import numpy as np
import time

from visual_servoing.msg import ConeLocation, ParkingError
from ackermann_msgs.msg import AckermannDriveStamped

class ParkingController():
    """
    A controller for parking in front of a cone.
    Listens for a relative cone location and publishes control commands.
    Can be used in the simulator and on the real robot.
    """
    def __init__(self):
        rospy.Subscriber("/relative_cone", ConeLocation,
            self.relative_cone_callback)

        DRIVE_TOPIC = rospy.get_param("~drive_topic") # set in launch file; different for simulator vs racecar
        self.drive_pub = rospy.Publisher(DRIVE_TOPIC,AckermannDriveStamped, queue_size=10)
        self.error_pub = rospy.Publisher("/parking_error",ParkingError, queue_size=10)

        self.parking_distance = .75 # meters; try playing with this number!
        self.max_velocity = 0.5
        self.max_steering_angle = 0.34
        self.relative_x = 0
        self.relative_y = 0
        
        #PID controller variables
        self.Kp_angle = 1.
        self.Kd_angle = 0.
        self.Kp_velocity = 1.
        self.Kd_velocity = 0.
        self.prev_time = None
        self.prev_angle = None
        self.prev_distance_error = None
        
        #Reversing parameters
        self.reverse_proportion = 1.
        self.reverse_mode = False
        self.angle_theshold = 0.1
        self.distance_threshold = 0.1
        
        #Drive outputs
        self.drive_angle = 0
        self.drive_velocity = 0

    def relative_cone_callback(self, msg):
        #Update PID parameters
        self.Kp_angle = rospy.get_param("Kp_angle", 1.)
        self.Kd_angle = rospy.get_param("Kd_angle", 0.)
        self.Kp_velocity = rospy.get_param("Kp_velocity", 1.)
        self.Kd_velocity = rospy.get_param("Kd_velocity", 0.)
        
        #Update reverse parameters
        self.reverse_proportion = rospy.get_param("reverse_proportion", 0.5)
        self.angle_theshold = rospy.get_param("angle_threshold", 0.1)
        self.distance_threshold = rospy.get_param("distance_threshold", 0.1)
        
        #Forward/back distance
        # + = ahead, - = behind
        self.relative_x = msg.x_pos
        #Right/left distance
        # + = left, - = right
        self.relative_y = msg.y_pos
        
        #Distance to cone
        relative_xy_array = np.array([self.relative_x,self.relative_y])
        relative_distance = np.linalg.norm(relative_xy_array)
        
        #Distance offset of target to robot - = too close, + = too far
        self.distance_error = relative_distance - self.parking_distance
        if self.relative_x < 0 and self.distance_error > 0:
            self.distance_error = self.distance_error * -1
            
        #Angle of cone to the robot. - = to the right, + = to the left
        relative_angle = np.arctan(self.relative_y/self.relative_x)
        
        #If we are at the right distance but the angle is way off, back up and try again
        if abs(self.distance_error) < self.distance_threshold and abs(relative_angle) > self.angle_theshold:
            self.reverse_mode = True
        
        # #Print for debugging
        # print("angle", abs(relative_angle))
        # print("distance", abs(self.distance_error))
        # print("reverse mode", self.reverse_mode)
        
        #When not reversing, run regular drive code
        if not self.reverse_mode:
            #Update PID controllers for distance and angle
            current_time = time.time()
            if self.prev_angle is not None and self.prev_distance_error is not None and self.prev_time is not None:
                delta_t = current_time - self.prev_time
                D_angle = (relative_angle - self.prev_angle) / delta_t
                D_velocity = (self.distance_error - self.prev_distance_error) / delta_t
            else:
                D_angle = 0
                D_velocity = 0
            self.prev_time = current_time
            self.prev_angle = relative_angle
            self.prev_distance_error = self.distance_error
            angle_out = (self.Kp_angle * relative_angle) + (self.Kd_angle * D_angle)
            velocity_out = (self.Kp_velocity * self.distance_error) + (self.Kd_velocity * D_velocity)
            
            #Cap velocity and send it out
            self.drive_velocity = np.clip(velocity_out, -self.max_velocity, self.max_velocity)
            #Cap angle and send it out. If backing up, use negative of angle
            self.drive_angle = np.clip(angle_out, -self.max_steering_angle, self.max_steering_angle)*np.sign(self.drive_velocity)
        
        else: #Reverse the car a little bit so it can try again
            #Straighten out and back up until robot is a distance away from the target (proportional to angle)
            if self.distance_error < self.reverse_proportion*(1+np.absolute(relative_angle)):
                self.drive_velocity = -self.max_velocity
                self.drive_angle = 0
            else:
                self.reverse_mode = False
        #Send error values   
        self.error_publisher()
        #Send drive values
        self.drive_command()
        
    def drive_command(self):
        drive_cmd = AckermannDriveStamped()

        #Header info
        drive_cmd.header.stamp = rospy.get_rostime()
        drive_cmd.header.frame_id = 'base_link'

        #Drive info
        drive_cmd.drive.steering_angle=self.drive_angle
        drive_cmd.drive.steering_angle_velocity=0.0
        drive_cmd.drive.speed=self.drive_velocity
        drive_cmd.drive.acceleration=0.0
        drive_cmd.drive.jerk=0.0

        self.drive_pub.publish(drive_cmd)
        
    def error_publisher(self):
        """
        Publish the error between the car and the cone. We will view this
        with rqt_plot to plot the success of the controller
        """
        error_msg = ParkingError()

        error_msg.x_error = self.relative_x
        error_msg.y_error = self.relative_y
        error_msg.distance_error = self.distance_error
        
        self.error_pub.publish(error_msg)

if __name__ == '__main__':
    try:
        rospy.init_node('ParkingController', anonymous=True)
        ParkingController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
