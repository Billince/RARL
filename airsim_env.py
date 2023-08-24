import time
import numpy as np
import airsim
import config
import pprint
import logging
import sys

import gym
from gym import spaces
from gym.utils import seeding
from gym.spaces import Box
from gym.spaces.box import Box
import math
import matplotlib.pyplot as plt
#import seaborn as sns
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)

clockspeed = 1
timeslice = 4 / clockspeed
floorZ= -0.5
outX = 4
outY = -0.2
speed_limit = 0.2
ACTION = ['00', '+x', '+y', '+z', '-x', '-y', '-z']
logger = logging.getLogger(__name__)

class Env:
    def __init__(self):
        # connect to the AirSim simulator
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.action_size = 3
        self.level = 0

        goal= self.client.simGetObjectPose('Boat2')

        self.episodeN = 0
        self.stepN = 0
        self.distance = np.sqrt(np.power((goal.position.x_val),2) + np.power((goal.position.y_val),2))
        #self.allLogs['distance'] = [self.distance]
        #self.allLogs['distance'] = [self.distance]
        
    def reset(self):
        self.level = 0
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # my takeoff
        self.client.simPause(False)
        self.client.moveByVelocityAsync(0, 0, -1, 2 * timeslice).join()
        self.client.moveByVelocityAsync(0, 0, 0, 0.1 * timeslice).join()
        self.client.hoverAsync().join()
        self.client.simPause(True)

        #start position for drone
        start_position= self.client.simGetObjectPose('PlayerStart_1')  

        # get object pose
        goal= self.client.simGetObjectPose('Boat2')
        self.client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(start_position.position.x_val, start_position.position.y_val, start_position.position.z_val), airsim.to_quaternion(10,10,10)), True)# Set Vehicle Pose to start
        #time.sleep(2)

        goal= self.client.simGetObjectPose('Boat2')

        distance = np.sqrt(np.power((goal.position.x_val),2) + np.power((goal.position.y_val),2))
        #self.allLogs = { 'reward': [0] }
        self.allLogs['distance'] = [distance]
        #self.allLogs['action'] = [1]

        #self.distance = np.sqrt(np.power((goal.position.x_val),2) + np.power((goal.position.y_val),2)+ np.power((goal.position.z_val),2))
        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        responses = self.client.simGetImages([airsim.ImageRequest(1, airsim.ImageType.DepthVis, True)])
        quad_vel = np.array([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val])
        observation = [responses, quad_vel]
        return observation

    def step(self, quad_offset):
        # move with given velocity
        quad_offset = [float(i) for i in quad_offset]
        #quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        self.client.simPause(False)

        
        goal= self.client.simGetObjectPose('Boat2')

        has_collided = False
        #landed = False
        self.client.moveByVelocityAsync(quad_offset[0], quad_offset[1], quad_offset[2], timeslice)
        #self.client.moveByVelocityAsync(quad_vel.x_val+quad_offset[0], quad_vel.y_val+quad_offset[1], quad_vel.z_val+quad_offset[2], timeslice)
        collision_count = 0
        start_time = time.time()


        while time.time() - start_time < timeslice:
            # get quadrotor states
            quad_pos = self.client.getMultirotorState().kinematics_estimated.position
            quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
            
            # decide whether collision occured
            collided = self.client.simGetCollisionInfo().has_collided

            collision = collided #or landed
            if collided == True:
                done = True
                reward = -100.0
                distance = np.sqrt(np.power((goal.position.x_val-quad_pos.x_val),2) + np.power((goal.position.y_val-quad_pos.y_val),2)+ np.power((goal.position.z_val-quad_pos.z_val),2))
       
            else: 
                done = False
                reward, distance = self.compute_reward(quad_pos)
        
            # Youuuuu made it
            if distance < 3:
                done = True
                reward = 100.0
                with open("reached.txt", "a") as myfile:
                    myfile.write(str(self.episodeN) + ", ")
            
            #self.addToLog('reward', reward)
            #rewardSum = np.sum(self.allLogs['reward'])
            self.addToLog('distance', distance)

        print('Distance %d Reward %.2f:' % (distance, reward))
        #print('Step %d Action %s Reward %.2f Info %s:' % (timestep, real_action, reward, info['status']))
        info = {"x_pos" : quad_pos.x_val, "y_pos" : quad_pos.y_val}

        #observe with depth camera 
        responses = self.client.simGetImages([airsim.ImageRequest(1, airsim.ImageType.DepthVis, True)])

        # get quadrotor states
        quad_pos = self.client.getMultirotorState().kinematics_estimated.position
        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity

        # decide whether done
        dead = has_collided #or quad_pos.y_val <= outY 
        done = dead #or quad_pos.y_val <= goal.position.y_val 

        quad_vel = np.array([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val])
        observation = [responses, quad_vel]

        return observation, reward, done, info
        # compute reward
       # reward = self.compute_reward(quad_pos,quad_vel,dead)

    def addToLog (self, key, value):
         if key not in self.allLogs:
             self.allLogs[key] = []
         self.allLogs[key].append(value)

    def compute_reward(self, quad_pos):

        goal= self.client.simGetObjectPose('Boat2')

        distance_now = np.sqrt(np.power((goal.position.x_val-quad_pos.x_val),2) + np.power((goal.position.y_val-quad_pos.y_val),2)+ np.power((goal.position.z_val-quad_pos.z_val),2))
        
        distance_before = self.allLogs['distance'][-1]
              
        r = -1
        
        
        r = r + (distance_before - distance_now)
            
        return r, distance_now

    def disconnect(self):
        self.client.enableApiControl(False)
        self.client.armDisarm(False)
        print('Disconnected.')
