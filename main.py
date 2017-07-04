# -*- coding: utf-8 -*-
"""
Created on Tue May 30 11:49:48 2017

@author: Wouter Eijlander, s4243242
"""
import numpy as np
import time
import random as r
import naoqi
import cv2
import multiprocessing
import Queue
import pickle
import wave

from ScaleVisual import *
from ScaleAudio import *
from ScaleProp import *

# Robot IPs. First four are NAO robots, and are currently only used for testing
#ip = "192.168.1.143" # Job
#ip = "192.168.1.102" # Naomi
#ip = "192.168.1.138" # Marvin
#ip = "192.168.1.102" # Jarvis
ip = "192.168.1.115" # Pepper
port = 9559

'''
IMPORTANT: CHANGE RECORDING NUMBER FOR EACH RECORDING THAT IS MADE; OTHERWISE IT WILL KEEP OVERWRITING THE SAME SOUND FILE.
'''

recording = 1
rate = 20
record_time = 40

def writeData(videoQueue,jointQueue,video_data,joint_data):
	try:
		# attempt to get a video frame from the video queue
		frame = videoQueue.get()
		
	except Queue.Empty: 
		# if there is no video frame, do nothing. Getting this multiple times indicates queue writing issues.
		print "No video available."
		frame = []
		
	try:
		# attempt to get joint positions from the joint queue
		joints = jointQueue.get()
		
	except Queue.empty: 
		# if there is no joint data, do nothing. Getting this multiple times indicates queue writing issues.
		print "no Joint data available"
		joints = [0,0,0,0,0,0]
	
	video_data.append(frame)
	joint_data.append(joints)

def requestVideo(videoQueue):
	'''
	Request the joints from the visual perception-object, and put them in the Queue
	This function is called as a process and is meant to ensure the images read from the Queue are very recent
	'''
	visual = ScaleVisual(ip, port, rate)
	while True:
		# Save an image		
		image = visual.getFrame()
		
		# if there is an image in the queue, replace it. This minimizes memory load which would otherwise increase quickly
		try:
			previous = videoQueue.get(False)
			del previous
		except Queue.Empty:
			pass
			
		videoQueue.put(image)

def requestJoints(jointQueue):
	'''
	Request the joints from the proprioception-object, and put them in the Queue. 
	This function is called as a process and is meant to ensure the joint states read from the Queue are very recent
	'''
	prop = ScaleProp(ip, port, rate)
	while True:
		joints = prop.getJoints()
		# if there is a joint state in the queue, replace it. This minimizes memory load which would otherwise increase quickly
		try:
			previous = jointQueue.get(False)
			del previous
		except Queue.Empty:
			pass
			
		jointQueue.put(joints)

if __name__ == "__main__":
	pythonBroker = naoqi.ALBroker("pythonBroker", "0.0.0.0", 9600, ip, port)
	print 'starting'

	out_file = open("Data/recording_{}".format(recording),"w")
	
	try:
		# initialize queues and all writer- and reader processes.
		videoQueue = multiprocessing.Queue()
		jointQueue = multiprocessing.Queue()
		
		recordAudio = ScaleAudio(recording)
		recordVideo = multiprocessing.Process(name='video_proc', target=requestVideo, args=(videoQueue,))
		recordJoints = multiprocessing.Process(name='joint_proc', target=requestJoints, args=(jointQueue,))

		# lists for storing both the video data and joint data
		video_data = list()
		joint_data = list()

		# start all three processes
		recordVideo.start()
		recordJoints.start()
		
		# run the script for approx 40 seconds, terminating when the exception is thrown.
		t = time.clock()
		while t < 40:
			recordAudio.record(0.05)
			writeData(videoQueue, jointQueue, video_data, joint_data)
			t= time.clock()

		# terminate all processes
		recordVideo.terminate()
		recordJoints.terminate()
		
		print "Video data size: ", np.shape(video_data), "\nJoint data size: ", np.shape(joint_data)
		
		# save recorded .wav file
		recordAudio.writeSound()
		
		# store video and joint data lists in a tuple
		recorded_data = (video_data, joint_data)
		pickle.dump(recorded_data, out_file)
		out_file.close()
		
	except KeyboardInterrupt: 
		# user interrupts script to stop it; terminates running processes.
		print "Keyboard pressed, terminating"
		recordVideo.terminate()
		recordJoints.terminate()
		
		print "Video data size: ", np.shape(video_data), "\nJoint data size: ", np.shape(joint_data)
		
		# save recorded .wav file
		recordAudio.writeSound()
		
		# store video and joint data lists in a tuple. Consider replacing this with a simple shut-down on all local proxies, for this to serve as an 'abort' button.
		recorded_data = (video_data, joint_data)
		pickle.dump(recorded_data, out_file)
		out_file.close()