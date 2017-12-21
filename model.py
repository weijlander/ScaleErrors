import scipy.io as io
import numpy as np
import math
import FREAK
import cv2
import naoqi
from ScaleVisual import *
from testBehaviour import *
from ScaleAudio import *
from scipy.special import expit
from PIL import Image

class Model(naoqi.ALModule):
	def __init__(self):
		self.ip		= "192.168.1.137"
		#self.ip 		= "192.168.1.143" # Job
		self.port 	= 9559
		self.rec		= "door_small_1"
		
		# Set the amount trained- either fully trained, or still performing errors.
		self.type = "trained"
		#self.type = "error"
		
		# Set number of categories we're using
		#self.ncats	= 2
		self.ncats	= 4
		
		self.looping = True
		self.nloops = 50
		
		self.version = "online"		# This is for testing in the lab with a NAO
		#self.version = "offline"			# This is for testing at home with no robot present. requires testfeats/train_descriptors.txt 
		
		self.filename 	= "robotexperiment/"+str(self.ncats)+"categories/"+str(self.type)+"/"+str(self.rec)
		self.file 			= open(self.filename, "w")
		
		input_dir_2 	= "ModelData/model/" 
		input_dir_4	= "ModelData/model4/"
		se_dir 			= "during_scale_errors/"
		se_mat_2 		= "model10.mat"
		se_mat_4		= "model16.mat"
		tr_dir 			= "aftertraining/"
		tr_mat 			= "model50.mat"
		
		if self.version == "online":
			self.objects 			= ["Chair","Door","Ball","Cylinder"]
			self.pythonBroker 	= naoqi.ALBroker("pythonBroker", "0.0.0.0", 9600, self.ip, self.port)
			self.motionProxy	= naoqi.ALProxy("ALMotion", self.ip, self.port)
			self.speechProxy	= naoqi.ALProxy("ALTextToSpeech", self.ip, self.port)
			self.postureProxy 	= naoqi.ALProxy("ALRobotPosture", self.ip, self.port)
			self.behaviour		= Behaviours(self.ip, self.port)
			self.listener 			= ScaleWords(self.ip,self.port,self.objects,name="listener")
			self.viewer 			= ScaleVisual(self.ip,self.port,30)
	
		if self.type=="trained":
			if self.ncats ==2:
				model_int_50	= io.loadmat(input_dir_2+tr_dir+tr_mat)
				model_aud		= io.loadmat(input_dir_2+"m_lang.mat", struct_as_record=False)
				model_act		= io.loadmat(input_dir_2+"m_action.mat", struct_as_record=False)
				model_v			= io.loadmat(input_dir_2+"m_percept.mat", struct_as_record=False)
			else:
				model_int_50	= io.loadmat(input_dir_4+tr_dir+tr_mat)
				model_aud		= io.loadmat(input_dir_4+"m_lang.mat", struct_as_record=False)
				model_act		= io.loadmat(input_dir_4+"m_action.mat", struct_as_record=False)
				model_v			= io.loadmat(input_dir_4+"m_percept.mat", struct_as_record=False)
			self.model_int	= model_int_50['model']
			
		elif self.type=="error":
			if self.ncats ==2:
				model_int_10	= io.loadmat(input_dir_2+se_dir+se_mat_2)
				model_v			= io.loadmat(input_dir_2+"m_percept.mat", struct_as_record=False)
				model_act		= io.loadmat(input_dir_2+"m_action.mat", struct_as_record=False)
				model_aud		= io.loadmat(input_dir_2+"m_lang.mat", struct_as_record=False)
			else:
				model_int_10	= io.loadmat(input_dir_4+se_dir+se_mat_4)
				model_v			= io.loadmat(input_dir_4+"m_percept.mat", struct_as_record=False)
				model_act		= io.loadmat(input_dir_4+"m_action.mat", struct_as_record=False)
				model_aud		= io.loadmat(input_dir_4+"m_lang.mat", struct_as_record=False)
			self.model_int	= model_int_10['model']
			
		weights_dec1	= self.model_int[0][0][0]
		weights_dec2	= self.model_int[0][0][2]
		weights_dec3	= self.model_int[0][0][4]
		b_dec			= self.model_int[0][0][6]
		c1_dec			= self.model_int[0][0][1]	
		c2_dec 			= self.model_int[0][0][3]
		c3_dec 			= self.model_int[0][0][5]
			
		self.model_v_1		= model_v['m_percept'][0][0][0][0]
		self.model_v_2		= model_v['m_percept'][1][0][0][0]
		self.model_v_3		= model_v['m_percept'][2][0][0][0]
		self.model_act_1		= model_act['m_action'][0][0]
		self.model_aud_1	= model_aud['m_lang'][0][0]
		
		# separate the weights for the visual layers, the proprioceptive layer, the auditive layer, and the integrations
		weights_v_1 	= self.model_v_1.W
		weights_v_2 	= self.model_v_2.W
		weights_v_3 	= self.model_v_3.W
		weights_act	= self.model_act_1.W
		weights_aud	= self.model_aud_1.W
		
		b_v_1 			= self.model_v_1.b
		b_v_2 			= self.model_v_2.b
		b_v_3 			= self.model_v_3.b
		b_act				= self.model_act_1.b
		b_aud			= self.model_aud_1.b
		
		c_v_1 			= self.model_v_1.c
		c_v_2 			= self.model_v_2.c
		c_v_3 			= self.model_v_3.c
		c_act				= self.model_act_1.c
		c_aud			= self.model_aud_1.c
		
		# three visual input layers
		self.visual_1 		= RBM(np.shape(weights_v_1), W=weights_v_1, B=b_v_1, c=c_v_1)
		self.visual_2 		= RBM(np.shape(weights_v_2), W=weights_v_2, B=b_v_2, c=c_v_2)
		self.visual_3 		= RBM(np.shape(weights_v_3), W=weights_v_3, B=b_v_3, c=c_v_3)
		
		# one action input layer, and one audio
		self.act_1				= RBM(np.shape(weights_act), W=weights_act, 	B=b_act, 	c=c_act)
		self.audit_1			= RBM(np.shape(weights_aud), W=weights_aud, 	B=b_aud, 	c=c_aud)
		
		# one integrator on top
		self.int_layer		= integrator(np.shape(weights_dec1), W1=weights_dec1, W2=weights_dec2, W3=weights_dec3,B=b_dec, c1=c1_dec, c2=c2_dec, c3=c3_dec)
		
	def make_decision(self):
		size 		= (360,240)
		
		try:
			if self.version == "online":
				# ensure the right posture
				self.postureProxy.goToPosture("Stand",0.8)
				self.motionProxy.setAngles(["HeadPitch","HeadYaw"],[0.25,r.uniform(-0.1,0.1)], 0.3)
				time.sleep(1)
				
				# take visual input and extract features, make random action input, and spot for a word for 5 seconds
				im 				= cv2.resize(self.viewer.getFrame(),size)
				Image.fromarray(im).show()
				in_act 			= np.random.normal(0.1, 0.05, size = 8)
				in_aud			= self.listener.wordSpot()
				in_vis 			= FREAK.calc_freak(im,size)
				in_vis			= FREAK.convertBinary(in_vis)
				
				if self.looping:					
					for k in range(self.nloops):
						# Decide probabilities for each action, and make that a binary vector
						visual,action,audio = self.perform_cycle(in_vis, in_act, in_aud)
						in_act 	= action
						in_aud 	= audio
						
						if np.amax(action)>0.5:
							action[action==np.amax(action)]=1
							action[action!=1]=0
							action = [int(dec) for dec in action[0]]
						else:
							action = [0 for each in range(8)]
						
					# transcribe the action vector to the number corresponding to that action.
					act_transcribed = self.transcribe(action)
					print "Action", act_transcribed, "was selected, based on model output:", in_act, "with audio input: ", in_aud
					
					# perform the action with the transcribed number
					self.save_data(in_vis,in_aud,in_act,action)
					self.perform_action(act_transcribed, in_act)
					self.motionProxy.rest()
					
				else:
					# Decide probabilities for each action, and make that a binary vector
					visual,action,audio = self.perform_cycle(in_vis, in_act, in_aud)
					output = action
					if np.amax(action)>0.5:
						action[action==np.amax(action)]=1
						action[action!=1]=0
						action = [int(dec) for dec in action[0]]
					else:
						action = [0 for each in range(8)]
					
					# transcribe the action vector to the number corresponding to that action.
					act_transcribed = self.transcribe(action)
					print "Action", act_transcribed, "was selected, based on model output:", output, "with audio input: ", in_aud
					
					# perform the action with the transcribed number
					self.save_data(in_vis,in_aud,output,action)
					self.perform_action(act_transcribed, output)
					self.motionProxy.rest()
					
			elif self.version=="offline":
				# read all FREAK descriptors previously determined
				with open("testfeats/train_descriptors.txt") as f:
					features = f.read().splitlines()
					
				decisions 	= []
				categories	= []
				actions		= []
				
				# for each feature, do the following:
				for x,feature in enumerate(features):
					# load the feature and its category, and make a random action and word
					category = int(feature.split(' ')[-1])
					if self.looping:					
					
						# 2 categories case is different, it should only take chairs and doors
						if self.ncats==2:
							if category in [1,2,3,4]:
								categories.append(category)
								in_vis 		= np.array(feature.split(' ')[:-1],dtype=np.dtype(int))
								in_act 		= np.random.normal(0.1, 0.05, size = 8)
								in_aud 		= np.random.normal(0.1, 0.05, size = 4)
								in_vis 		= np.reshape(in_vis, (1,np.shape(in_vis)[0]))
								in_act		= np.reshape(in_act, (1,np.shape(in_act)[0]))
								in_aud		= np.reshape(in_aud, (1,np.shape(in_aud)[0]))
								for k in range(self.nloops):									
									# determine what action is best given current input, and make the vector binary
									visual,action,audio = self.perform_cycle(in_vis, in_act, in_aud)
									in_act = action
									in_aud = audio
								actions.append(action)
								
						# otherwise it's the 4 category case, which can just go through all features indiscriminately
						else:
							categories.append(category)
							in_vis 		= np.array(feature.split(' ')[:-1],dtype=np.dtype(int))
							in_act 		= np.random.normal(0.1, 0.05, size = 8)
							in_aud 		= np.random.normal(0.1, 0.05, size = 4)
							in_vis 		= np.reshape(in_vis, (1,np.shape(in_vis)[0]))
							in_act		= np.reshape(in_act, (1,np.shape(in_act)[0]))
							in_aud		= np.reshape(in_aud, (1,np.shape(in_aud)[0]))
							for k in range(self.nloops):
								# determine what action is best given current input, and make the vector binary
								visual,action,audio = self.perform_cycle(in_vis, in_act, in_aud)
								in_act = action
								in_aud = audio
							actions.append(action)
							
					else:
						# 2 categories case is different, it should only take chairs and doors
						if self.ncats==2:
							if category in [1,2,3,4]:							
								categories.append(category)
								in_vis 		= np.array(feature.split(' ')[:-1],dtype=np.dtype(int))
								in_act 		= np.random.normal(0.1, 0.05, size = 8)
								in_aud 		= np.random.normal(0.1, 0.05, size = 4)
								
								# determine what action is best given current input, and make the vector binary
								visual,action,audio = self.perform_cycle(in_vis, in_act, in_aud)
								actions.append(action)
						# otherwise it's the 4 category case, which can just go through all features indiscriminately
						else:					
							categories.append(category)
							in_vis 		= np.array(feature.split(' ')[:-1],dtype=np.dtype(int))
							in_act 		= np.random.normal(0.1, 0.05, size = 8)
							in_aud 		= np.random.normal(0.1, 0.05, size = 4)
					
							# determine what action is best given current input, and make the vector binary
							visual,action,audio = self.perform_cycle(in_vis, in_act, in_aud)
							actions.append(action)
							
				for action in actions:
					if np.amax(action)>0.5:
						action[action==np.amax(action)]=1
						action[action!=1]=0
						action = [int(dec) for dec in action[0]]
					else:
						action = [0 for each in range(8)]
					decisions.append(action)
				
				# determine all action numbers corresponding to the action vectors
				decs_transcribed = [self.transcribe(dec) for dec in decisions]
				
				correct = 0
				scalemistakes = 0
				undecided = 0
				
				# loop over all decisions...
				for x,dec in enumerate(decs_transcribed):
					# check if they match the correct decision for that input
					if dec==categories[x]:
						correct+=1
						
					# and if it doesn't, check if it was a scale error
					if self.ncats==2:
						if (dec==3 and categories[x]==4) or (dec==4 and categories[x]==3) or (dec==1 and categories[x]==2) or (dec==2 and categories[x]==1):
							scalemistakes+=1
					else:
						if (dec==3 and categories[x]==4) or (dec==4 and categories[x]==3) or (dec==1 and categories[x]==2) or (dec==2 and categories[x]==1) or (dec==7 and categories[x]==8) or (dec==8 and categories[x]==7) or (dec==5 and categories[x]==6) or (dec==6 and categories[x]==5):
							scalemistakes+=1
					
					# else, if there was no action at all
					if dec ==0:
						undecided += 1
				
				n_errors = len(categories)-correct
				
				print "With ", self.ncats, " categories, and the ", self.type, "network.\n"
				
				print "Accuracy:", float(correct)/len(categories)
				print "Total errors: ", float(n_errors)/len(categories)
				print "Scale errors: ", float(scalemistakes)/len(categories)
				print "Category errors: ", float(n_errors-scalemistakes-undecided)/len(categories)
				print "Undecided: ", float(undecided)/len(categories)
				
		except KeyboardInterrupt: 
			# user interrupts script to stop it; terminates running processes.
			print "Keyboard pressed, terminating"
			self.motionProxy.stopMove()
			self.postureProxy.goToPosture('Crouch')
			self.pythonBroker.shutdown()
			self.file.close()
			
	def transcribe(self,dec):
		if 1 in dec:
			i = dec.index(1)
			if i==0: # small cylinder
				return 8
			elif i==1: # small ball
				return 6
			elif i==2: # small chair
				return 2
			elif i==3: # small door
				return 4
			elif i==4: # large cylinder
				return 7
			elif i==5: # large ball
				return 5
			elif i==6: # large chair
				return 1
			elif i==7: # large door
				return 3
		else:
			return 0

	def perform_action(self,action,output):
		if action == 1:
			bla = action
			#print "big chair, based on model output: ", output
			#self.behaviour.useBigChair()
		elif action == 2:
			bla = action
			#print "small chair, based on model output: ", output
			#self.behaviour.useSmallChair()
		elif action == 3:
			bla = action
			#print "big door, based on model output: ", output
			#self.behaviour.useBigDoor()
		elif action == 4:
			bla = action
			#print "small door, based on model output: ", output
			#self.behaviour.useSmallDoor()
		elif action == 5:
			bla = action
			#print "Large ball spotted, based on model output: ", output
			self.speechProxy.say("This is a arge ball!")
		elif action == 6:
			bla = action
			#print "Small ball spotted, based on model output: ", output
			self.speechProxy.say("That is a small ball!")
		elif action == 7:
			bla = action
			#print "Large cylinder spotted, based on model output: ", output
			self.speechProxy.say("This is a large cylinder!")
		elif action == 8:
			bla = action
			#print "Small cylinder spotted, based on model output: ", output
			self.speechProxy.say("That is a small cylinder!")
		else:
			print "No decision could be made. ", output

	def perform_cycle(self, in_vis, in_act, in_aud):
		encoding = self.upward_pass(in_vis, in_act, in_aud)
		vis, act, aud = self.downward_pass(encoding)
		return vis, act, aud

	def upward_pass(self, x_visual, x_act, x_aud):
		y_visual		= self.visual_1.up_pass(x_visual)
		y_visual		= self.visual_2.up_pass(y_visual)
		y_visual		= self.visual_3.up_pass(y_visual)
		y_act			= self.act_1.up_pass(x_act)
		y_aud		= self.audit_1.up_pass(x_aud)
		y_int 		= self.int_layer.up_pass(y_visual,y_act,y_aud)
		return y_int

	def downward_pass(self, y_int):
		y_visual, y_act, y_aud 	= self.int_layer.down_pass(y_int)
		x_aud							= self.audit_1.down_pass(y_aud)
		x_act								= self.act_1.down_pass(y_act)
		y_visual 						= self.visual_3.down_pass(y_visual)
		y_visual 						= self.visual_2.down_pass(y_visual)
		x_visual 						= self.visual_1.down_pass(y_visual)
		return x_visual, x_act, x_aud
	
	def save_data(self, freak, word, output, action):
		self.file.write("word: {}".format(word))
		self.file.write("action: {}".format(action))
		self.file.write("output: {}".format(output))
		self.file.write("FREAK: {}".format(freak))
		self.file.close()
	
class RBM():
	def __init__(self, (in_size, out_size), W=None, B=None, c=None):
		self.in_size = in_size
		self.out_size = out_size
		if W.shape == (in_size,out_size):
			self.W = W
		else:
			raise ValueError('shape of W should match (in_size, out_size).')	
		if B.transpose().shape == (out_size, 1):
			self.B = B
		else:
			raise ValueError('shape of bias_input should match (in_size, 1).')
		self.c = c
		
	def up_pass(self, x):
		y = expit(self.B + np.dot(x,self.W))
		return y
		
	def down_pass(self, y):
		x = expit(self.c + np.dot(y,self.W.transpose()))
		return x

class integrator():
	def __init__(self, (in_size, out_size), W1, W2, W3, B, c1, c2, c3):
		self.in_size = in_size
		self.out_size = out_size
		if W1.shape == (in_size,out_size):
			self.W1 = W1
			self.W2 = W2
			self.W3 = W3
		else:
			raise ValueError('shape of W should match (in_size, out_size).')
		if B.transpose().shape == (out_size, 1):
			self.B = B
		else:
			raise ValueError('shape of bias_input should match (in_size, 1).')
		self.c1 = c1
		self.c2 = c2
		self.c3 = c3
	
	def up_pass(self, x_visual, x_prop, x_aud):
		s_vis 	= np.dot(x_visual,self.W1)
		s_prop	= np.dot(x_prop, self.W2)
		s_aud	= np.dot(x_aud, self.W3)
		return expit(self.B + s_vis + s_prop + s_aud)
		
	def down_pass(self, x_down):
		d_visual 	= expit(self.c1 + np.dot(x_down,self.W1.transpose()))
		d_prop 		= expit(self.c2 + np.dot(x_down,self.W2.transpose()))
		d_aud 		= expit(self.c3 + np.dot(x_down,self.W3.transpose()))
		return d_visual, d_prop, d_aud

model = Model()
model.make_decision()
model.postureProxy.goToPosture("Crouch",0.5)
model.motionProxy.rest()