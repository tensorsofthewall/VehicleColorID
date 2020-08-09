# Code for vehicle color identification using YOLO with a kNN color classifier 
# using histogram to measure RGB levels.

# 
#

#Importing libraries
import cv2 as cv2
import argparse
import sys
import numpy as np
import os
import color_feature_extractor as feature_extractor
import knn as knn


# Functions to allow augmentation of YOLO bounding boxes with color classification

# Draws prediction boxes with labels for the output image

def drawPredBoxes(frame,classID,confidence, left, top, right, bottom, prediction):
	cv2.rectangle(frame,(left,top),(right,bottom),(255,180,90),4)
	label = '%.2f' % confidence
	if yolo_classes:
		assert(classID < len(yolo_classes))
		label = '%s:%s Color=%s' % (yolo_classes[classID],label,prediction)

	#To display color, class and confidence score at top of bounding box
	labelTextSize, baseLine = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
	top = max(top, labelTextSize[1])
	cv2.rectangle(frame,(left,top-round(1.5*labelTextSize[1])),(left+round(1.5*labelTextSize[0]),top+baseLine),(255,255,255),cv2.FILLED)
	cv2.putText(frame,label,(left,top),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)



# Function to apply confidence thresholding and NMS suppression on bounding boxes
# before running kNN classifier for color classification

def color_id_process(frame,outputs):
	frame_height = frame.shape[0]
	frame_width = frame.shape[1]

	classIDs = []
	confidences = []
	boxes = []

	#Removing bounding boxes with low confidence scores
	for out in outputs:
		for detection in out:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			if(confidence>conf_thresh):
				center_x = int(detection[0]*frame_width)
				center_y = int(detection[1]*frame_height)
				width = int(detection[2]*frame_width)
				height = int(detection[3]*frame_height)
				left = int(center_x-width/2)
				top  = int(center_y-height/2)
				classIDs.append(classID)
				confidences.append(float(confidence))
				boxes.append([left,top,width,height])

	#NMS Suppression to eliminate overlapping bounding boxes 
	indices = cv2.dnn.NMSBoxes(boxes,confidences,conf_thresh,nms_thresh)
	for i in indices:
		i = i[0]
		box = boxes[i]
		left, top, width, height = box
		if left<0:
			left=0
		if top<0:
			top=0
		obj_bb_cropped = frame[top:top+height, left:left+width]
		feature_extractor.test_img(obj_bb_cropped)
		prediction = knn.predict(training_data, test_data)
		drawPredBoxes(frame,classIDs[i],confidences[i],left,top,left+width,top+height,prediction)


# Function to get output layer names from network
def getOutputLayers(net):
	#Get names of all layers in network
	layerNames = net.getLayerNames()
	#To return only outputs layers i.e unconnected outputs
	return [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]



#Parsing user arguments
parser = argparse.ArgumentParser(description='Vehicle Color Identification using YOLO ')
parser.add_argument('--img',help='Path to single image')
parser.add_argument('--video',help='Path to video file')
args = parser.parse_args()


#Initialization of parameters
conf_thresh = 0.5 #Confidence Threshold for bounding box retention
nms_thresh = 0.4 #Non-maximal suppression threshold for removing overlapping bounding boxes
input_width = 416
input_height = 416

#Data paths
training_data = './training.data'
test_data = './test_img.data'
yolo_classes_file = './darknet/data/coco.names'
yolo_classes = None
with open(yolo_classes_file,'rt') as f:
	yolo_classes=f.read().rstrip('\n').split('\n')

yolo_config = './darknet/cfg/yolov3.cfg'
yolo_weights = './darknet/yolov3.weights'

output_dir = './outputs/'
if not os.path.isdir(output_dir):
	os.mkdir(output_dir)



#Initializing YOLO Darknet model
yolo = cv2.dnn.readNetFromDarknet(yolo_config,yolo_weights)
yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

if args.img:
	if not os.path.isfile(args.img):
		print("Input image file doesn't exist\n")
		sys.exit(1)
	cap = cv2.VideoCapture(args.img)
	outputFile = args.img[:-4]+'_yolo_out_py.jpg'
	outputFile = outputFile.split('/')[-1]
elif args.video:
	if not os.path.isfile(args.video):
		print("Input video file doesn't exist\n")
		sys.exit(1)
	cap = cv2.VideoCapture(args.video)
	outputFile = args.video[:-4]+'_yolo_out_py.avi'
	outputFile = outputFile.split('/')[-1]
else:
	print("Invalid arguments\n")
	sys.exit(1)

#intialize video writer for saving video
if args.video:
	vid_writer = cv2.VideoWriter(os.path.join(output_dir,outputFile), cv2.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))


while cv2.waitKey(1) < 0:
	# if not args.img_fol:
	ret, frame = cap.read()

	#Check if end of video or no image
	if not ret:
		print("Completed, output is saved in output folder\n")
		cv2.waitKey(3000)
		cap.release()
		break
	#OpenCV model needs blob image for input
	blob_img = cv2.dnn.blobFromImage(frame,1/255,(input_width,input_height),[0,0,0],1,crop=False)
	yolo.setInput(blob_img)
	outputs = yolo.forward(getOutputLayers(yolo))

	#Processing for color classification
	color_id_process(frame, outputs)
	t, _ = yolo.getPerfProfile()
	label = 'Inference time: %.2f ms' %(t*1000.0/cv2.getTickFrequency())
	cv2.putText(frame,label,(0,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))

	if(args.img):
		cv2.imwrite(os.path.join(output_dir,outputFile),frame.astype(np.uint8))
	else:
		vid_writer.write(frame.astype(np.uint8))