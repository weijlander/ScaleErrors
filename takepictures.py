import numpy as np
import scipy.io as io
import time
import random as r
import naoqi
import cv2
import multiprocessing
import Queue
import pickle
import wave
from PIL import Image

from ScaleVisual import *
from ScaleAudio import *
from ScaleProp import *

ip = "192.168.1.137" # Marvin
port = 9559

recording = "ball_small"
n_pics = 50

def getFrame(videoProxy, cam):
	'''
	Function that records the video data and writes to the queue. Should be called as a process.
	'''
	# Save an image
	image_container = videoProxy.getImageRemote(cam)
			
	# get image width and height
	width = image_container [0]
	height = image_container [1]
		
	# the 6th element contains the pixel data
	values = map(ord, list( image_container [6]))
		
	# Write the pixel values to the image array for later use
	image = np.array(values , np.uint8 ).reshape((height, width))
	image = np.asarray(image)
	return image

def subscribeCam(videoProxy):
	# Ensure no duplicate camera connections are active on the robot. This improves program stability.
	cams = videoProxy.getSubscribers()
	for cam in cams:
		videoProxy.unsubscribe(cam)

	# initialize some variables for the camera
	cam_name = "camera"     	# creates an identifier for the camera subscription
	cam_type = 0            			# 0 for top camera , 1 for bottom camera
	res = 0                 				# 0 = 160x120, 1 = 320x240, 2 = 640x480, 3 = 1280x960, 4 = 2560x1920, 5 = 1280x720, 6 = 1920x1080. NAO only support 0 through 3.
	col_space = 0          			# luma colorspace, thus 1 channel
	fps = 30               				# the requested frames per second. with res = 0,1,2 this can be up to 30. with others it is max 7.

	#subscribing a camera returns a string identifier to be used later on.
	cam = videoProxy.subscribeCamera (cam_name , cam_type , res , col_space , fps)
	return cam

videoProxy = naoqi.ALProxy("ALVideoDevice", ip, port)
motionProxy = naoqi.ALProxy("ALMotion", ip, port)
postureProxy = naoqi.ALProxy("ALRobotPosture", ip, port)


motionProxy.wakeUp()
postureProxy.goToPosture("Stand",0.5)
motionProxy.setBreathEnabled('Legs',True)
motionProxy.setBreathConfig([["BPM",10],["Amplitude",0.8]])
#motionProxy.setBreathEnabled('Head',True)

for i in range(n_pics):
	motionProxy.setAngles(["HeadPitch","HeadYaw"],[0.25,r.uniform(-0.1,0.1)], 0.3)
	time.sleep(1)
	file_name = "VisualData/{0}{1}.jpg".format(recording,i)
	cam = subscribeCam(videoProxy)
	im = getFrame(videoProxy, cam)
	print np.shape(im)
	image = Image.fromarray(im)
	if i == 0:
		image.show()
	image.save(file_name)
	

motionProxy.setBreathEnabled('Body',False)
postureProxy.goToPosture("Crouch",0.5)