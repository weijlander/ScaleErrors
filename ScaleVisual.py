import numpy as np
import time
import naoqi
import cv2
import multiprocessing
import Queue

class ScaleVisual(naoqi.ALModule):
	'''
	Class for subscribing to a camera and writing the data to a queue in set intervals. 
	'''
	def __init__(self, videoProxy, rate):
		self.videoProxy = videoProxy#naoqi.ALProxy("ALVideoDevice", ip, port)
		self.rate = rate

		# Ensure no duplicate camera connections are active on the robot. This improves program stability.
		cams = self.videoProxy.getSubscribers()
		for cam in cams:
			self.videoProxy.unsubscribe(cam)

		# initialize some variables for the camera
		cam_name = "camera"     	# creates an identifier for the camera subscription
		cam_type = 0            		# 0 for top camera , 1 for bottom camera
		res = 0                 				# 0 = 160x120, 1 = 320x240, 2 = 640x480, 3 = 1280x960, 4 = 2560x1920, 5 = 1280x720, 6 = 1920x1080. NAO only support 0 through 3.
		col_space = 0          			# luma colorspace, thus 1 channel
		fps = 30               				# the requested frames per second. with res = 0,1,2 this can be up to 30. with others it is max 7.
		
		#subscribing a camera returns a string identifier to be used later on.
		self.cam = self.videoProxy.subscribeCamera (cam_name , cam_type , res , col_space , fps)
	
	def __init__(self, ip, port, rate):
		self.videoProxy = naoqi.ALProxy("ALVideoDevice", ip, port)
		self.rate = rate

		# Ensure no duplicate camera connections are active on the robot. This improves program stability.
		cams = self.videoProxy.getSubscribers()
		for cam in cams:
			self.videoProxy.unsubscribe(cam)

		# initialize some variables for the camera
		cam_name = "camera"     	# creates an identifier for the camera subscription
		cam_type = 0            		# 0 for top camera , 1 for bottom camera
		res = 0                 				# 0 = 160x120, 1 = 320x240, 2 = 640x480, 3 = 1280x960, 4 = 2560x1920, 5 = 1280x720, 6 = 1920x1080. NAO only support 0 through 3.
		col_space = 0          			# luma colorspace, thus 1 channel
		fps = 30               				# the requested frames per second. with res = 0,1,2 this can be up to 30. with others it is max 7.
		
		#subscribing a camera returns a string identifier to be used later on.
		self.cam = self.videoProxy.subscribeCamera (cam_name , cam_type , res , col_space , fps)
	
	def getFrame(self):
		'''
		Function that records the video data and writes to the queue. Should be called as a process.
		'''
		# Save an image
		image_container = self.videoProxy.getImageRemote(self.cam)
			
		# get image width and height
		width = image_container [0]
		height = image_container [1]
		
		# the 6th element contains the pixel data
		values = map(ord, list( image_container [6]))
		
		# Write the pixel values to the image array for later use
		image = np.array(values , np.uint8 ).reshape((height, width, 1))
		return image
		