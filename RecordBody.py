# -*- coding: utf-8 -*-
"""
Created on Tue May 30 11:49:48 2017

@author: Wouter Eijlander, s4243242
"""
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
import FREAK

from ScaleVisual import *
from ScaleAudio import *
from ScaleProp import *
from ReadData import *
from TestBehaviour import *

# Robot IPs. First four are NAO robots, and are currently only used for testing
#ip = "192.168.1.143" # Job
#ip = "192.168.1.102" # Naomi
ip = "192.168.1.137" # Marvin
#ip = "192.168.1.102" # Jarvis
#ip = "192.168.1.115" # Pepper
port = 9559

'''
IMPORTANT: CHANGE RECORDN NUMBER FOR EACH RECORDING THAT IS MADE; OTHERWISE IT WILL KEEP OVERWRITING THE SAME FILES.
'''

recordn = 1
recordtype = "door_small{}"
recording = recordtype.format(recordn)

record_time = 40
n_joint_r = 25
rate = 20

def writeData(jointQueue,joint_data):
	try:
		# attempt to get joint positions from the joint queue
		joints = jointQueue.get()
		joint_data.append(joints)
		
	except Queue.empty: 
		# if there is no joint data, do nothing. Getting this multiple times indicates queue writing issues.
		print "no Joint data available"
		pass

def requestVideo():
	'''
	Request the joints from the visual perception-object, and put them in the Queue
	This function is called as a process and is meant to ensure the images read from the Queue are very recent
	'''
	visual = ScaleVisual(ip, port, rate)
	# Save an image		
	image = visual.getFrame()
	
	return image

def requestBehaviour(startQueue,ip,port):
	'''
	Request for a certain behaviour to be called.
	'''
	behaviour = Behaviours(ip, port)
	if "door" in recording:
		if "small" in recording:
			behaviour.useSmallDoor(startQueue)
		else:
			behaviour.useBigDoor(startQueue)
	elif "chair" in recording:
		if "small" in recording:
			behaviour.useSmallChair(startQueue)
		else:
			behaviour.useBigChair(startQueue)
	if "ball" in recording:
		if "small" in recording:
			behaviour.useSmallBall(startQueue)
		else:
			behaviour.useBigBall(startQueue)
	if "cylinder" in recording:
		if "small" in recording:
			behaviour.useSmallCylinder(startQueue)
		else:
			behaviour.useBigCylinder(startQueue)
		
def requestJoints(jointQueue,startQueue):
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
		
		# check if the movement has started and if the movement hasn't ended yet
		if (not startQueue.empty()):
			jointQueue.put(joints)
		time.sleep(1)

if __name__ == "__main__":
	pythonBroker 	= naoqi.ALBroker("pythonBroker", "0.0.0.0", 9600, ip, port)
	print 'starting'
	file_name 		= "InputData/recording_" + recording
	out_file 			= open(file_name,"w")
	frame_size 	= (360,240)
	postureProxy = naoqi.ALProxy("ALRobotPosture",ip,port)
	
	try:
		# initialize queues and all writer- and reader processes.
		videoQueue 	= multiprocessing.Queue()
		jointQueue 	= multiprocessing.Queue()
		startQueue 	= multiprocessing.Queue()
		endQueue		= multiprocessing.Queue()
		
		wordlist 		= ["chair","door","ball","cylinder"]
		listener 				= ScaleWords(ip,port,wordlist)
		#recordAudio 			= ScaleAudio(recording)
		
		recordJoints 			= multiprocessing.Process(name='joint_proc', target=requestJoints, args=(jointQueue,startQueue))
		performBehaviour 	= multiprocessing.Process(name='behav_proc', target=requestBehaviour, args=(startQueue,ip,port))

		# lists for storing both the video data and joint data
		video_data 		= list()
		joint_data 		= list()
		word_data 		= list()

		# take a snapshot from the camera before the movement
		postureProxy.goToPosture("Stand",0.8)
		time.sleep(1)
		frame 			= cv2.resize(requestVideo(),frame_size)
		word_data 	= listener.wordSpot()
		
		# start all processes
		recordJoints.start()
		performBehaviour.start()
		
		# run the script for record_time seconds
		t = time.clock()
		while t < record_time:
			#recordAudio.record(1)
			writeData(jointQueue, joint_data)
			t= time.clock()
		print "end of movement indicated."
		
		# terminate all processes
		recordJoints.terminate()
		
		video_data = frame
		video_data = FREAK.calc_freak(frame, size)
		video_data = FREAK.convertBinary(video_data)
		
		print "Video data size: ", np.shape(video_data), "\nJoint data size: ", np.shape(joint_data), "\nWord data size: ", np.shape(word_data)
		
		while len(joint_data)<n_joint_r:
			joint_data.append(joint_data[-1])
		joint_data = joint_data[:n_joint_r]
		
		print "\nJoint data size after clipping: ", np.shape(joint_data)
		
		# save recorded .wav file
		#recordAudio.writeSound()
		
		# store video and joint data lists in a tuple. Consider replacing this with a simple shut-down on all local proxies, for this to serve as an 'abort' button.
		recorded_data = (frame, video_data, joint_data, word_data)
		pickle.dump(recorded_data, out_file)
		io.savemat(file_name+'.mat', mdict={'arr':recorded_data})
		out_file.close()
		output = readData(file_name)
		
		performBehaviour.terminate()
		pythonBroker.shutdown()
		
	except KeyboardInterrupt: 
		# user interrupts script to stop it; terminates running processes.
		print "Keyboard pressed, terminating"
		
		recordJoints.terminate()
		
		video_data = frame
		video_data = FREAK.calc_freak(frame, size)
		video_data = FREAK.convertBinary(video_data)
		
		print "Video data size: ", np.shape(video_data), "\nJoint data size: ", np.shape(joint_data), "\nWord data size: ", np.shape(word_data)
		
		while len(joint_data)<n_joint_r:
			joint_data.append(joint_data[-1])
		joint_data = joint_data[:n_joint_r]
		
		print "\nJoint data size after clipping: ", np.shape(joint_data)
		
		# save recorded .wav file
		#recordAudio.writeSound()
		
		# store video and joint data lists in a tuple. Consider replacing this with a simple shut-down on all local proxies, for this to serve as an 'abort' button.
		recorded_data = (frame, video_data, joint_data, word_data)
		pickle.dump(recorded_data, out_file)
		io.savemat(file_name+'.mat', mdict={'arr':recorded_data})
		out_file.close()
		output = readData(file_name)
		
		performBehaviour.terminate()
		pythonBroker.shutdown()