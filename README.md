# T-RexNet - A Hardware-Aware Neural Network for Real-Time Detection of Small Moving Objects

## Introduction
This repository contains a demo of T-RexNet for the specific task of tennis ball tracking.
The pretrained network is provided for easy usage.
The program takes in input the video of a tennis match and outputs a new video where a bounding box is drawn around the detected tennis ball.
The full paper can be found at https://www.mdpi.com/1424-8220/21/4/1252/pdf.

## Requirements
The author used Python, version 3.6.9, and the TensorFlow library,version 1.14.
Newer versions may work but it is not guaranteed.
The paths below must be included in the interpreter paths:
[tensorflowCheckoutPath]/models/research/object_detection/models
[tensorflowCheckoutPath]/models/research
[tensorflowCheckoutPath]/models/research/object_detection

## Usage
Just clone this repository and substitute the input video with the video you want to perform tennis ball tracking on.
For best performance, given the way T-RexNet works, the camera has to be static.

## Reference
Please cite this work as Canepa, Alessio, et al. "T-RexNet—A Hardware-Aware Neural Network for Real-Time Detection of Small Moving Objects." Sensors 21.4 (2021): 1252.
