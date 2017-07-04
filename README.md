"# ScaleErrors" 

Running main.py runs a recording of video data, joint data, and audio data (.wav). 
Line 21 through 35 contain some parameters for recording data:
	ip 				Ip address of the robot. There's several options, uncomment the desired robot, comment the rest.
	port				port through which the robots connect. Do not change this.
	recording		name of the recording. Change this for each new recording, otherwise previous recordings will be lost
	rate				maximum framerate for recording. Maximum value is 30, but generally, recordings will achieve no higher than 12 fps.
	record_time	maximum time to record in seconds. the script will record this amount of time, unless interrupted.

After setting these parameters, simply run the script, and recording starts. 
If the goal has been reached, press any button on the keyboard to exit the script and save recordings.