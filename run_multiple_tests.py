import os
import cv2
import argparse

parser = argparse.ArgumentParser(description='Vehicle Color Identification using YOLO for multiple images ')
parser.add_argument('--img_path',help='Path to folder containing images')
parser.add_argument('--vid_path',help='Path to folder containing videos')

args = parser.parse_args()

if args.img_path:
	imgs = os.listdir(args.img_path)
	for img in imgs:
		command = "python VehicleColorID.py --img " + os.path.join(args.img_path,img)
		print(command)
		os.system(command)
elif args.vid_path:
	vids = os.listdir(args.vid_path)
	for vid in vids:
		command = "python VehicleColorID.py --video "+os.path.join(args.vid_path.vid)
		print(command)
		os.system(command)