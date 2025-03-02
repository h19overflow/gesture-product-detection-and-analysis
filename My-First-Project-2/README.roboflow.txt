
My First Project - v2 2025-02-21 9:55pm
==============================

This dataset was exported via roboflow.com on February 21, 2025 at 1:56 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 324 images.
Objects are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise, upside-down
* Random shear of between -15° to +15° horizontally and -13° to +13° vertically
* Random brigthness adjustment of between -22 and +22 percent
* Random Gaussian blur of between 0 and 2.5 pixels
* Salt and pepper noise was applied to 1.72 percent of pixels


