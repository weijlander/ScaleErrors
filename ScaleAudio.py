import numpy as np
import time
import random as r
import naoqi
import cv2
import multiprocessing
import Queue
import pickle
import wave
import pyaudio

class ScaleWords(naoqi.ALModule):
	'''
	Class used to detect specific spoken words
	'''
	def __init__(self, ip, port, wordlist, name="listener"):
		try:
			p = naoqi.ALProxy(name)
			p.exit()
		except: 
			pass
		naoqi.ALModule.__init__(self, name);
		
		self.name					=name
		self.wordlist 				= wordlist
		self.speecher				= naoqi.ALProxy("ALSpeechRecognition",ip, port)
		self.am 						= naoqi.ALProxy("ALAutonomousMoves")
		
		self.am.setExpressiveListeningEnabled(False)
		self.am.setBackgroundStrategy("none")
		self.speecher.setLanguage("English")
		self.speecher.setVocabulary(self.wordlist, True)
	
		self.memory= naoqi.ALProxy("ALMemory")
		self.word = " "
		self.done = False
		
	def wordSpot(self):
		'''
		Function that spots for any words in the given list
		'''
		self.memory.subscribeToEvent("WordRecognized", self.name, "getWord")
		time.sleep(2)
		self.memory.unsubscribeToEvent("WordRecognized", self.name)
		self.speecher.pause(True)
		return self.vectorise()
	
	def getWord(self, key, value, msg):
		'''
		Sets the spotted word value
		'''
		if not self.done:
			found = value[0].split(' ')[1]
			self.word = found
			self.done = True
		
	def vectorise(self):
		vector = np.zeros(4)
		if self.word in self.wordlist:
			index = self.wordlist.index(self.word)
			vector[index] = 1
		vector = [int(num) for num in vector]
		return np.array(vector, dtype=np.dtype(int))

class ScaleAudio(naoqi.ALModule):
	'''
	Class used for recording audio and writing to a .wav file.
	'''
	def __init__(self, recording):
		# initalize file name and make .wav file.
		self.filename = "InputData/recording_{0}_audio.wav".format(recording)
		self.audio_file = wave.open(self.filename, 'wb')
		self.name = "recordAudio"
		
		# intialize some parameters used for recording
		self.FORMAT = pyaudio.paInt16
		self.CHANNELS = 2
		self.RATE = 44100
		self.CHUNK = 1024
		self.RECORD_SECONDS = 0.1
		self.frames = []

		# open the input audio stream
		self.audio = pyaudio.PyAudio()
		self.stream = self.audio.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)
		
		# set the parameters for the .wav file
		self.audio_file.setnchannels(self.CHANNELS)
		self.audio_file.setsampwidth(self.audio.get_sample_size(self.FORMAT))
		self.audio_file.setframerate(self.RATE)
	
	def record(self, time):
		'''
		Function that records audio data to the internal sound frames. 
		'''
		for i in range(0, int(self.RATE / self.CHUNK * time)):
			data = self.stream.read(self.CHUNK)
			self.frames.append(data)
		
	def writeSound(self):
		'''	
		Function to close all data streams and save recorded audio to a .wav file.
		'''
		self.stream.stop_stream()
		self.audio.terminate()
		self.stream.close()
		self.audio_file.writeframes(b''.join(self.frames))
		self.audio_file.close()
		
if __name__=="__main__":
	#ip = "192.168.1.143" # Job
	#ip = "192.168.1.102" # Naomi
	ip = "192.168.1.137" # Marvin
	#ip = "192.168.1.102" # Jarvis
	#ip = "192.168.1.115" # Pepper
	port = 9559
	pythonBroker = naoqi.ALBroker("pythonBroker", "0.0.0.0", 9600, ip, port)
	words = ["Chair","Door","Ball","Cylinder"]
	listener = ScaleWords(ip,port,words)
	print listener.wordSpot()