import time
import random as r
import naoqi

# Robot IPs. First four are NAO robots, and are currently only used for testing
#ip = "192.168.1.143" # Job
#ip = "192.168.1.102" # Naomi
#ip = "192.168.1.137" # Marvin
#ip = "192.168.1.102" # Jarvis
#ip = "192.168.1.115" # Pepper
#port = 9559

class Behaviours(naoqi.ALModule):
	def __init__(self,ip,port):
		self.ip = ip
		self.port = port
		self.pythonBroker = naoqi.ALBroker("pythonBroker", "0.0.0.0", 9600, ip, port)
		self.motionProxy = naoqi.ALProxy("ALMotion", ip, port)
		self.postureProxy = naoqi.ALProxy("ALRobotPosture",ip,port)
		self.speechProxy = naoqi.ALProxy("ALTextToSpeech",ip,port)
	
	def walk50(self):
		# Makes the nao walk forward 50cm
		x 			= [0.5] # forward movement speeds
		y 			= [0.0] # sideways movement speeds
		theta 	= [0.0] # rotational speeds
		
		# move forward
		self.motionProxy.moveToward(x[0],y[0],theta[0])
		time.sleep(10) # using x = 0.5, y = 0, this moves nao forward by 50cm
		self.motionProxy.stopMove()

	def useBigDoor(self):
		# Assumes Nao is 60 cm away from the door, directly in front of it
		# Makes the nao walk trhough the door and look around as it does
		# Still to include: prevent nao from falling over if accidentally used for a small door
		x 			= [0.7] # forward movement speeds
		y 			= [0.0] # sideways movement speeds
		theta 	= [0.0] # rotational speeds
		
		# Initialize parameters for the gesture
		names = ["HeadPitch","HeadYaw"]
		angles = [[-0.25,-0.25,-0.25,-0.25,0.4,-0.4,0.4,-0.4,0],[0.4,0.4,-0.4,-0.4,0,0,0,0,0],]
		times = [[2,3,5,6,8,9,10,11,12] for each in names]
		
		self.postureProxy.goToPosture("Stand", 0.5)
		self.walk50()
		self.motionProxy.angleInterpolation(names,angles,times,True)
		self.walk50()
		self.postureProxy.goToPosture("Crouch",0.8)
		
	def useSmallDoor(self):
		# Assumes Nao is 60 cm away form the door, directly inn front of it
		# 
		x 			= [0.7] # forward movement speeds
		y 			= [0.0] # sideways movement speeds
		theta 	= [0.0] # rotational speeds
		
		# Initialize parameters for the gesture
		names = ["HeadPitch","HeadYaw"]
		angles = [[-0.1,-0.1,-0.1,-0.1,0,0,0,0,0],[0.4,0.4,-0.4,-0.4,0.2,-0.2,0.2,-0.2,0],]
		times = [[2,3,5,6,8,9,10,11,12] for each in names]
		
		self.postureProxy.goToPosture("Stand", 0.5)
		self.walk50()
		self.postureProxy.goToPosture("Crouch",0.8)
		self.motionProxy.angleInterpolation(names,angles,times,True)
		
		# turn around
		self.postureProxy.goToPosture("Stand", 0.5)
		self.motionProxy.moveToward(x[0],y[0],theta[0])
		time.sleep(8) 
		self.motionProxy.stopMove()
		self.postureProxy.goToPosture("Crouch",0.8)
		
	def useBigChair(self):
		# Assumes Nao is 50cm away from the large chair, crouched.
		# Makes Nao walk up to the chair and turn around, ready to sit backwards.
		x 			= [0.5, 0.0, -0.5] # forward movement speed
		y 			= [0.0, 0.0, 0.0] # sideways movement speed
		theta 	= [0.0, 0.5, 0.0] # rotational speed, values above 0 turn nao to the left, values below 0 turn nao to the right.
		
		self.postureProxy.goToPosture("Stand", 0.5)
		self.walk50()
		
		# turn around
		self.motionProxy.moveToward(x[1],y[1],theta[1])
		time.sleep(8) 
		self.motionProxy.stopMove()
		
		# walk slightly backwards toward the chair
		self.motionProxy.moveToward(x[2],y[2],theta[2])
		time.sleep(3.3)
		self.motionProxy.stopMove()
		
		self.sitChair()

	def sitChair(self):
		# Makes the nao sit backwards ono a 10cm high chair. Quite a lot of potential code, thus a separate function
		# Assumes the nao is standing directly in front of the chair.
		# This makes the nao sit down almost perfectly, but triggers fall detection.
		self.motionProxy.setFallManagerEnabled(False)
		names = ["LShoulderPitch","RShoulderPitch","LShoulderRoll","RShoulderRoll","LElbowRoll","RElbowRoll","LKneePitch","RKneePitch","LHipPitch","RHipPitch","LAnklePitch","RAnklePitch"]
		angles = [[1.8,1.9,1.9,1.9,1.9],[1.8,1.9,1.9,1.9,1.9],[-0.3,-0.2,-0.2,0,0.2],[0.3,0.2,0.2,0,-0.2],[-0.1,-0.1,-0.4,-0.4,-0.7],[0.1,0.1,0.4,0.4,0.7],[0,0.8,1.5,1.4,1.2],[0,0.8,1.5,1.4,1.2],[0,-0.35,-0.5,-0.8,-1],[0,-0.35,-0.5,-0.8,-1],[0,-0.5,-0.8,-0.5,0],[0,-0.5,-0.8,-0.5,0]]
		times = [[2.0, 5.0, 8.0, 11.0, 14.0] for each in names]
		self.motionProxy.angleInterpolation(names,angles,times,True)
			
		self.postureProxy.goToPosture("SitOnChair",0.4)
		self.motionProxy.setFallManagerEnabled(True)
		self.motionProxy.rest()
		
	def useSmallChair(self):
		# Assumes nao is 50cm away from the small chair, crouched
		# Makes Nao walk up to the chair and doubt its use since it is too small
		
		x 			= [0.5] # forward movement speeds
		y 			= [0.0] # sideways movement speeds
		theta 	= [0.0] # rotational speeds
		
		# Initialize parameters for the gesture
		names = ["HeadPitch","HeadYaw", "RShoulderPitch", "RShoulderRoll", "RElbowRoll","RElbowYaw","RWristYaw","LShoulderPitch","LElbowYaw","LWristYaw"]
		angles = [[0.4,0.4,0.4,0.4,0.4,0.4,0.4],[0.4,-0.4,0.4,-0.4,0.4,-0.4,0],[-1,-1,-1,-1,-1,-1,1.5],[-0.6,-0.3,-0.6,-0.3,-0.6,-0.3,0],[1.5,1.5,1.5,1.5,1.5,1.5,0.1],[-0.3,-0.3,-0.3,-0.3,-0.3,-0.3,0],[1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.0,1.0,1.0,1.0,1.0,1.0,1.5],[-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,0],[-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,0]]
		times = [[2,3,4,5,6,7,9] for each in names]
		
		self.postureProxy.goToPosture("Stand", 0.5)
		self.walk50()
		self.speechProxy.setVolume(0.5)
		self.motionProxy.angleInterpolation(names,angles,times,True)
		self.speechProxy.say("This chair is too small!")
		self.postureProxy.goToPosture("Crouch",0.8)
