import cv2
from os import listdir
from os.path import isfile, join
import sys

def main():
	print sys.version, "\n"
	dataset_path 		= "testdata"
	out_file_name 	= "descriptors.txt"
	out_file 				= open(out_file_name, 'w')
	features 			= []
	encodings			= [] 
	size 					= (360,240) #resize to Beata's sizes
	
	# list all the files in the dataset
	files = [f for f in listdir(dataset_path) if isfile(join(dataset_path, f))]
	
	for x,file in enumerate(files):
		print "Analyzing file",x+1,"out of",len(files),"files."
		
		object_name = file.split("_")[0]						# checks the name of the object
		object_size = file.split("_")[1][:5]					# checks size of the object (small and large are both 5 letters, thus checking the first 5 letters in a string is fine)
		
		image = cv2.imread(dataset_path+"/"+file,0)	# open the current image in grayscale
		image = cv2.resize(image, size)
		if x == 0:
			cv2.imshow('image',image)
		
		encoding = encode(object_name,object_size)	# calculate the object encoding...
		encodings.append(encoding)							# ... and add it to the list of encodings
		
		feature_values = calc_freak(image,size)				# calculate the FREAK descriptor...
		features.append(feature_values)					# ... and add them to the list of features

		print feature_values

		out_file.write("{0}_{1}\n".format(feature_values,encoding))	

	
def calc_freak(im,size):
	keypoints = []
	tmp_values = []
	
	keypoints.append(cv2.KeyPoint(round(size[0]/2), round(size[1]/2),1)) # set keypoint with size 1 to middle of the image
	extractor = cv2.xfeatures2d.FREAK_create(False,False, 22.0) 	# make freak extractor
	
	# attempt feature extraction using the FREAK extractor...
	try:
		keypoints, descriptors = extractor.compute(im, keypoints)
	except:
		# ... and print the exception if that fails
		print "Unexpected error:", sys.exc_info()
		pass
	keypoints = None
	extractor = None
	
	# TODO: TURN INTO BINARY ENCODING
	
	return descriptors

def encode(name, size):
	# encodes the object based on the name and size. This function could be done far more beautifully using a dictionary, but this suffices for now
	if name=="door":
		if size=="large":
			return 1
		elif size=="small":
			return 2
	elif name=="chair":
		if size=="large":
			return 3
		elif size=="small":
			return 4

main()