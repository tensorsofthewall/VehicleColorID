# VehicleColorID
Performs object segmentation using YOLOv3 with color recognition using color histograms and kNN classifier.

## Libraries required
1. OpenCV - 4.2
2. Pillow - 6.1
3. Numpy - 1.18.1
4. Matplotlib - 3.1.3

## Other Requirements
- Darknet and YOLOv3 cfg and weights installed. Follow instructions on the [official website](https://pjreddie.com/darknet/yolo/) to install Darknet.
- color_feature_extractor and knn modules need to be placed in the same folder as VehicleColorID file.
- Dataset of colors to be detected should be placed with these files.

## Training Data Creation
Run color_feature_extractor.py using the following command: python color_feature_extractor.py --path $COLOR_DATASET_PATH$

## Running the classifier
YOLOv3 pretrained on COCO dataset is used as the object detector, and the color-based kNN classifier predicts the color of the detected object.
- For single image file: python VehicleColorID.py --video $video_file_path.extension$
- For single video file: python VehicleColorID.py --img $img_file_path.extension$
- For multiple images or videos:
  - Group such files in a single folder for execution
  - Run command: python run_multiple_tests.py --img_path $Image_Folder_Path$ **or** python run_multiple_tests.py --vid_path $Video_Folder_Path$

Results will be saved in the 'outputs' directory.
