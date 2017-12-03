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

class Model():
	def __init__(self):
		self.ip		= "192.168.1.137"
		self.port 	= 9559
		
		# Set the amount trained- either fully trained, or still performing errors.
		self.type = "trained"
		#self.type = "error"
		
		#self.version = "online"		# This is for testing in the lab with a NAO
		self.version = "offline"			# This is for testing at home with no robot present. requires testfeats/descriptor.txt 
		
		weights_file 	= "weights.mat"
		bias_file 		= "biases.mat"
		
		if self.version == "online":
			self.pythonBroker = naoqi.ALBroker("pythonBroker", "0.0.0.0", 9600, ip, port)
			self.videoProxy = naoqi.ALProxy("ALVideoDevice", ip, port)
		
			self.scaleVis 		= ScaleVisual(self.videoProxy, 30)
			self.scaleAud		= ScaleAudio(audioProxy)
		
		model_v					= io.loadmat("ModelData\model\m_percept.mat", struct_as_record=False)
		self.model_v_1		= model_v['m_percept'][0][0][0][0]
		self.model_v_2		= model_v['m_percept'][1][0][0][0]
		self.model_v_3		= model_v['m_percept'][2][0][0][0]

		model_act				= io.loadmat("ModelData\model\m_action.mat", struct_as_record=False)
		self.model_act_1		= model_act['m_action'][0][0]

		model_aud		= io.loadmat("ModelData\model\m_lang.mat", struct_as_record=False)
		self.model_aud_1	= model_aud['m_lang'][0][0]

		model_int_50	= io.loadmat("ModelData\model\model50.mat")
		self.model_int_50	= model_int_50['model']
		
		model_int_10	= io.loadmat("ModelData\model\model10.mat")
		self.model_int_10	= model_int_10['model']
		
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
		
		if self.type=="trained":
			weights_dec1	= self.model_int_50[0][0][0]
			weights_dec2	= self.model_int_50[0][0][2]
			weights_dec3	= self.model_int_50[0][0][4]
			b_dec			= self.model_int_50[0][0][6]
			c1_dec			= self.model_int_50[0][0][1]	
			c2_dec 			= self.model_int_50[0][0][3]
			c3_dec 			= self.model_int_50[0][0][5]
		elif self.type=="error":
			weights_dec 	= self.model_int_10[0][0][0]
			weights_dec2	= self.model_int_10[0][0][2]
			weights_dec3	= self.model_int_10[0][0][4]
			b_dec			= self.model_int_10[0][0][6]
			c1_dec			= self.model_int_10[0][0][1]	
			c2_dec 			= self.model_int_10[0][0][3]
			c3_dec 			= self.model_int_10[0][0][5]
		
		# three visual input layers
		self.visual_1 		= RBM(512, 256, W=weights_v_1, B=b_v_1, c=c_v_1)
		self.visual_2 		= RBM(256, 96, 	W=weights_v_2, B=b_v_2, c=c_v_2)
		self.visual_3 		= RBM(96, 32,  	W=weights_v_3, B=b_v_3, c=c_v_3)
		
		# one action input layer, and one audio
		self.act_1				= RBM(8, 32, W=weights_act, 	B=b_act, 	c=c_act)
		self.audit_1			= RBM(4, 32, W=weights_aud, 	B=b_aud, 	c=c_aud)
		
		# one integrator on top
		self.int_layer		= integrator(32,48, W1=weights_dec1, W2=weights_dec2, W3=weights_dec3,B=b_dec, c1=c1_dec, c2=c2_dec, c3=c3_dec)
		
	def make_decision(self):
		size 		= (360,240)
		if self.version == "online":
			im 				= cv2.resize(self.scaleVis.getFrame(),size)
			in_vis 			= FREAK.calc_freak(im,size)
			in_vis			= FREAK.convertBinary(in_vis)
		elif self.version=="offline":
			tempFeature 	= open("testfeats/descriptor.txt")
			in_vis 			= np.array([line.split(' ') for line in tempFeature][0][:-1],dtype=np.dtype(int))
		
		in_act 		= np.random.randint(2, size = 8)
		in_aud 		= [0,0,0,1] #WRITE WORDSPOTTING SCRIPT FOR THIS
		decisions 	= []	
		
		#while ((len(decisions)<2 or decisions[-1]!=decisions[-2]) and len(decisions)<20):
		action = self.perform_cycle(in_vis, in_act, in_aud)
		decisions.append(action)
		
		print decisions	
		#perform_action(act)
		
	def perform_action(self,action):
		if action == 1:
			useBigDoor(motionProxy)
		elif action == 2:
			useSmallDoor(motionProxy)
		elif action == 3:
			useBigChair(motionProxy)
		elif action == 4:
			useSmallChair(motionProxy)

	def perform_cycle(self, in_vis, in_act, in_aud):
		encoding = self.upward_pass(in_vis, in_act, in_aud)
		vis, act, aud = self.downward_pass(encoding)
		return act

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
	
class RBM():
	def __init__(self, in_size, out_size, W=None, B=None, c=None):
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
	def __init__(self, in_size, out_size, W1, W2, W3, B, c1, c2, c3):
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
		x_summed = []
			# assumed weights shape and allocation : (vis_out+prop_out+aud_out,dec_in)
		s_vis 	= np.dot(x_visual, self.W1)
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
