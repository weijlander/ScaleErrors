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

class ScaleAudio():
	'''
	Class used for recording audio and writing to a .wav file.
	'''
	def __init__(self, recording):
		# initalize file name and make .wav file.
		self.filename = "Data/recording_{0}_audio.wav".format(recording)
		self.audio_file = wave.open(self.filename, 'wb')
		
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
		