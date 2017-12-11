import numpy as np
import time
import naoqi
import cv2
import multiprocessing
import Queue
import random as r

class ScaleProp(naoqi.ALModule):
	'''
	Class for subscribing to the arm joints and writing the data to a queue in set intervals. 
	'''
	def __init__(self, ip, port, rate):
		
		self.motionProxy = naoqi.ALProxy("ALMotion", ip, port)
		self.rate = rate
		
		# Set data regarding the joints, and how quickly they should relax.
		self.names = ['RShoulderPitch','RShoulderRoll','RElbowYaw','RElbowRoll','RWristYaw','RHand','LShoulderPitch','LShoulderRoll','LElbowYaw','LElbowRoll','LWristYaw','LHand','RHipYawPitch','RHipPitch','RKneePitch','RAnklePitch','RHipRoll','RAnkleRoll','LHipYawPitch','LHipPitch','LKneePitch','LAnklePitch','LHipRoll','LAnkleRoll']
		stiffnesses = [[0.0,1.0] for each in range(len(self.names))]
		times = [[0.5,1.0] for each in range(len(self.names))]
		self.motionProxy.setAngles(["HeadPitch","HeadYaw"],[0.25,r.uniform(-0.1,0.1)], 0.3)
		
		# Smoothly remove stiffness on the right arm. This is better for Pepper than simply using motionProxy.rest()
		self.motionProxy.stiffnessInterpolation(self.names, stiffnesses, times)
			
	def getJoints(self):
		'''
		Function that records proprioceptive data and writes to a queue. Should be called as a process.
		'''
		joints = self.motionProxy.getAngles(self.names, False)
		return joints