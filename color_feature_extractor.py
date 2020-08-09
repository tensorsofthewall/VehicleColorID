from PIL import Image
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse

def test_img(img):
  colors = ('b','g','r')
  img_channels = cv2.split(img)
  features = []
  features_data = []
  channel_counter=0
  for (img_channel, color) in zip(img_channels,colors):
    channel_counter+=1
    histogram = cv2.calcHist([img_channel],[0],None,[256],[0,256])
    features.extend(histogram)
    features_data.append(str(np.argmax(histogram)))
    if channel_counter==3:
      features_data = ",".join(features_data)

  with open('test_img.data','w') as f:
    f.write(features_data)


def training_img_histogram(img_path):
  colors = ('b','g','r')
  if 'red' in img_path:  
    data_source = 'red'
  elif 'yellow' in img_path:
    data_source = 'yellow'
  elif 'black' in img_path:
    data_source = 'black'
  elif 'blue' in img_path:
    data_source = 'blue'
  elif 'green' in img_path:
    data_source = 'green'
  elif 'white' in img_path:
    data_source = 'white'
  elif 'violet' in img_path:
    data_source = 'violet'
  elif 'orange' in img_path:
    data_source = 'orange'
  elif 'brown' in img_path:
    data_source = 'brown'
  elif 'pink' in img_path:
    data_source = 'pink'

  features = []
  features_data = []
  img_data = cv2.imread(img_path)
  img_channels = cv2.split(img_data)
  channel_counter = 0
  for (img_channel, color) in zip(img_channels,colors):
    channel_counter+=1
    histogram = cv2.calcHist([img_channel],[0],None,[256],[0,256])
    features.extend(histogram)

    features_data.append(str(np.argmax(histogram)))
    if channel_counter==3:
      features_data = ",".join(features_data)

  with open('training.data','a') as f:
    f.write(features_data+','+data_source+'\n')

def color_data_training(dataset_path):
  for color_path in os.listdir(dataset_path):
    for f in os.listdir(os.path.join(dataset_path,color_path)):
      training_img_histogram(os.path.join(dataset_path,color_path,f))

if __name__=="__main__":
  parser = argparse.ArgumentParser(description='Training data generation\n')
  parser.add_argument('--path',help='Training data folder path')
  args = parser.parse_args()

  color_data_training(args.path)