"# ScaleErrors" 

Required python modules:
PyAudio 				https://people.csail.mit.edu/hubert/pyaudio/
cv2-contrib			http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html
naoqi					
numpy					
scipy

== List of scripts and their use ==
-FREAK.py
	Used in imports for calculating FREAK descriptors from an image- recommended size 360x240. Contains functions for calculating the descriptors and converting them to binary representations. Can be run individually to calculate these for images in a given subfolder, and will save the descriptors in a single .txt file called testfeats/train_descriptors.txt
-Model.py
	Used to run the model, either online using a NAO robot, or offline using previously saved FREAK descriptors in the subfolder testfeats. This is basically the project's 'main' script, and uses several of the other scripts including FREAK, ScaleProp and ScaleVisual. Will save its outputs to a .txt file containing the word input data, the seen FREAK descriptor, the action that it selected, and the raw action output. 4
-ReadData.py
	Used to read data recorded using RecordBody.py, which saves data using a pickler. Contains a function that returns the exact objects saved during the recording (described below)
-RecordBody.py
	Used to record the body states of an action. More concretely, it records:
		frame: A 2D vector representation of its visual input in grayscale.
		video_data: the FREAK descriptor resulting from frame.
		joint_data: 25 joint positions from all joints in the NAO's body.
		word_data: a binary vector of length 4 resulting from the wordspotting script (ball, cylinder, chair, door)
	The script saves this data using Pickle, which can be read using ReadData.py, as well as in a Matlab struct.
	Contains some legacy code used for recording input from the laptop's microphone (those in the robot aren't dependable) and saving as a .wav file. Was written on Windows, but doesn't work properly for Mac, and may not do so on a virtual machine either.
-ScaleAudio.py
	Used for all things related to audio processing. Contains two classes: ScaleAudio, which is used for recording audio from a microphone, and ScaleWords, which is used for NAO's in-built wordspotting capabilities.	
-ScaleProp.py
	Used for recording proprioceptive states. Simply contains a class that initializes a motionproxy with joints of interest, and has a function for getting those joints' positions.	
-ScaleVisual.py
	Used for dealing with the robot's visual functions. Contains a class that allows for pulling video frames from the top camera.
-TakePictures.py
	Used for recording visual input of a certain object type. This script can be run individually to have the robot perform its idle animation, and take n_pics pictures, saving each one as a separate .jpg file in the subfolder Visualdata/
	Starts with some parameters, the major ones being:
		n_pics: an integer indicating the number of pictures to be taken in the current session.
		recording: a string indicating the type of object (i.e. "ball_small")	
-TestBehaviour.py
	Contains a class that deals with behavioural output, containing functions that predefine behaviours related to certain objects.


== List of subfolders ==
-InputData
	Contains data recorded from running RecordBody.py.
-ModelData
	-model
		contains the model trained on 2 categories
	-model4
		contains the model trained on 4 categories	
-robotexperiment
	Contains output files saved by Model.py, separated into subfolders detailign number of categories, and the amount of training.
-testfeats
	Contains .txt files of FREAK descriptors. Running FREAK.py individually saves descriptors of all pictures in VisualData/ in the file train_descriptors.txt
-VisualData
	Contains pictures recorded using TakePictures.py


== General positioning instructions ==
For this project, there are some constants regarding the robot's positioning relative to the target objects and the researcher:
	- The mat on the floor is a slightly matte, dark grey colour, and has been laid down on the ground witht he edges slightly resting on the lab's cabinets.
	- The cabinets containing the LEGO crates are closed
	- The mat has three taped markings:
		- white tape with the inscription "Robot's toes", and a line. Ensure the robot's feet sensors overlap this line, and the robot's line of sight crosses the line.
		- small white tape approximately 35cm from the aforementioned tape. This marks where balls and cylinder should be placed as described below.
		- yellow-green tape approximately 70cm from the "robot's toes" tape. This marks where chairs and the goal door should be placed.
	- Balls and cylinders are placed on a small rubber circle on top of a large blue cylinder. Both of these can be found in boxes behind the cabinets in the lab. 
		Place the cylinder on top of the marking tape, and lay the rubber circle on top of the cylinder, black side upwards. 
		In the middle of this circle, place either cylinder upright, or either ball on top of a small lego tire (these are kept in the cabinets, but are currently also stored witht he balls). This prevents the balls from rolling off.
	- When recording data or running the model, the researcher must be in front of the robot, preferably just ouside the field of view. NAO's wordspotting is most accurate from the front, and downright doesn't work from behind due to microphone insensitivities. This also allows the researcher to intervene quickly if the robot falls down.
The chairs, balls and small cylinder are stored on top of the cabinet in the middle of the lab. The small cylinder is made of grey LEGO wheels. The balls are stored on top of LEGO tires, to prevent them rolling away. The large cylinder is the whiteboard marker, with the plastic side facing the robot. The doors are stored in the back of the lab, behind the couch. Look for a large wooden board, and two trapezoidal pieces of wood- these can be slid into the grooves of the main piece to allow it to stand on itself.


== Instructions for recording an action ==
Recording an action is done using RecordBody.py. 
Place the robot in Crouching position as indicated above. Place the target object as indicated above.
When running this script, ensure these match your wishes:
	ip: the robot's ip, there is a list of the ones we have, keep only one uncommented.
	recordn: The number of this type of recording you're making (i.e. 5 for the fifth small door recording)
	recordtype: A string pertaining to the type of recording you're making, ending with {}. recordings will be saved to a file with the name in recordtype, and the number in recordn.
	record_time: the time spent recording. default value is 40 seconds, which enables recording of all predefined actions.
After ensuring this (and changing the necessary parameters for successive recordings), run the script. The robot will stand up, and take a picture. It will then indicate it is listening ("boop-beep")- say what object it is seeing. Ensure you are not behind the robot, but arguably in front of it- otherwise NAO's native wordspotting won't work properly. The robot will start the movement that matches the parameter recordtype. Wait for it to finish, and return it to its starting position. 

== Instructions for running the model ==
Running the model is done using Model.py.
Place the robot in Crouching position as indicated above. Place the target object as indicated above.
When running the script, ensure the following:
	The used computer is connected to the correct network
	The parameter self.ip matches the used robot
	The parameter self.rec is the filename you want the outputs to be saved to
	The parameters self.type ('trained' or 'error'), self.ncats (2 or 4), self.looping (True or False), self.nloops (any integer), and self.version ('online' or 'offline' to match the presence or absence of a robot) match the type of model you want to run.
After checking these (and changing self.rec in subsequent model runs) run the script. The robot will stand up, and take a picture. It will then indicate it is listening ("boop-beep")- say what object it is seeing. Ensure you are not behind the robot, but arguably in front of it- otherwise NAO's native wordspotting won't work properly. The robot will take a few seconds to make a decision, and will then start the decided action. The python console used to run the program will also print some info on the decision. Wait for the robot to finish, and return it to its starting position.

== When the robot complains about hot motors ==
Press the chest button once (skip the robot's IP by pressing the button once again). If it indicates "some of my motors are getting hot", no immediate action is needed. If it indicates "Some of my motors are too hot", performance may be impaired. Allow the robot to rest by doing the following: Press the chest button twice. The robot will sigh soudly, indicating it has released its joints. Wait for 10-15 minutes.