import pickle
import wave
import scipy as sc

def readData(filename):
	'''
	Reads data from the recording number passed in the parameter. This is the same parameter as used during recording (i.e. a number, or a name. NOT the entire filename)
	Returns a tuple of (video_data[x,y,n_frames], joint_data[n_joints, n_frames], sound_data(sampling_rate,sound_data[n_samples]))
	'''
	
	(frame, video_data, joint_data, word_data) = pickle.load(open(filename))
	
	# this version returns a Wave_read object 
	sound_data = wave.open(filename+"_audio.wav",'r')
	return frame, video_data, joint_data, word_data, sound_data