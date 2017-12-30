import cv2
from os import listdir
from os.path import isfile, join
import sys
import numpy as np
import re

if __name__=='__main__':
	'''
	If the script is run on itself, it calculates FREAK descriptors and binarizes them, and saves them in one .txt file using specific formatting discussed earlier in the project (space-separated integers, with a category appended to the end.)
	'''
	print sys.version, "\n"
	dataset_path 		= "VisualData"
	out_file_name 	= "tesfeats\train_descriptors1.txt"
	out_file 				= open(out_file_name, 'w')
	features 			= []
	encodings			= []
	size 					= (81,72)
	
	# list all the files in the dataset
	files = [f for f in listdir(dataset_path) if isfile(join(dataset_path, f))]
	
	for x,file in enumerate(files):
		#if x ==0:
		print "Analyzing file",x+1,"out of",len(files),"files."
		print file
		object_name = file.split("_")[0]				# checks the name of the object
		object_size = file.split("_")[1][:5]			# checks size of the object (small and large are both 5 letters, thus checking the first 5 letters in a string is fine)
		
		image = cv2.imread(dataset_path+"/"+file,0)	# open the current image in grayscale
		image = cv2.resize(image,size)
				 
		encoding = encode(object_name,object_size)	# calculate the object encoding...
		encodings.append(encoding)						# ... and add it to the list of encodings
		
		feature_values = calc_freak(image,size)		# calculate the FREAK descriptor...
		features.append(feature_values)				# ... and add them to the list of features
		feature_values = convertBinary(feature_values.tolist())
		
		output = '{} {}\n'.format(['{}'.format(x) for x in feature_values],encoding)
		output = re.sub('[\[\',\'\]]','',output)		# remove all unnecessary characters from the saved string (commas, brackets etc.)
		out_file.write(output)


def convertBinary(features):
	# Converts a list of integers to a list of 0s and 1s representing the integers. 
	# Does not implement any splitting between binary representations, but all binaries are 8-bit encodings.
	binary_feats = []
	for feature in features:
		binary_feature = [int(x) for x in list('{0:08b}'.format(feature))] # format to binary in 8-bit using string formatting into a list of ints
		binary_feats.extend(binary_feature)			# add the ints to the binary features list
	return binary_feats


def calc_freak(im,size):
	'''
	Calculate the freak descriptor given an image and its size- this is not yet a 512-length binary vector!
	recommended size for this project is 360x240.
	'''
	keypoints = []
	descriptors = []
	
	keypoints.append(cv2.KeyPoint(int(np.ceil(float(size[0])/2)),int(np.ceil(float(size[1])/2)),1)) # set keypoint with size 1 to middle of the image
	extractor = cv2.xfeatures2d.FREAK_create(False,False,11.0) 	# make freak extractor
	
	# attempt feature extraction using the FREAK extractor...
	try:
		keypoints, descriptors = extractor.compute(im, keypoints)
	except:
		# ... and print the exception if that fails
		print "Unexpected error:", sys.exc_info()
		pass
	return descriptors[0]


def encode(name, size):
	# encodes the object based on the name and size. This function could be done far more beautifully using a dictionary, but this suffices for now
	if name=="chair":
		if size=="large":
			return 1
		elif size=="small":
			return 2
	elif name=="door":
		if size=="large":
			return 3
		elif size=="small":
			return 4
	elif name=="ball":
		if size=="large":
			return 5
		elif size=="small":
			return 6
	elif name=="cylinder":
		if size=="mediu":
			return 7
		elif size=="small":
			return 8																													
