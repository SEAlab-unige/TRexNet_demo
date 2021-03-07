import os
import tensorflow as tf
import cv2
import time
from os import listdir
from os.path import isfile, join
from utils import label_map_util
from utils import visualization_utils as vis_util
import numpy as np

frameNumberStart = 0
frameNumberEnd = 10000000

cap = cv2.VideoCapture('video/TennisAmateur4_cutted.mp4')
outPutVideo = cv2.VideoWriter('outputVideo.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 25, (1280,720))

MODEL_NAME = 'TRexNet_TennisBallTracking' #hawknet

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'networks/' + MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('annotations', 'label_map.pbtxt')

# Number of classes to detect
NUM_CLASSES = 1

buffer = []

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

frameNumber = 0
prev_image_original = None

# Detection
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
            ret, image_np = cap.read()
            frameNumber = frameNumber + 1
            if frameNumber < frameNumberStart:
                continue
            elif frameNumber > frameNumberEnd:
                break
            if image_np is None:
                break

            image_original = image_np.copy()
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

            image_np_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

            buffer.append(image_np_gray.copy())
            if len(buffer) == 3:
                image_np = np.stack((cv2.absdiff(buffer[2], buffer[1]), buffer[1], cv2.absdiff(buffer[1], buffer[0])),
                                    axis=2)
                buffer.pop(0)

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)

            # Extract image tensor
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Extract detection boxes
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Extract raw detection boxes
            raw_boxes = detection_graph.get_tensor_by_name('raw_detection_boxes:0')
            # Extract detection scores
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            # Extract detection classes
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            # Extract number of detections
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections, raw_boxes) = sess.run(
                [boxes, scores, classes, num_detections, raw_boxes],
                feed_dict={image_tensor: image_np_expanded})

            _, numBoxes = vis_util.visualize_boxes_and_labels_on_image_array(
                prev_image_original if prev_image_original is not None else image_original,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                skip_scores=False,
                skip_labels=True,
                max_boxes_to_draw=100,
                min_score_thresh=0.9,
                use_normalized_coordinates=True,
                line_thickness=1,
                backgroundWeights=None)

            # Display output
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            cv2.imshow('object detection', prev_image_original if prev_image_original is not None else image_original)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            outPutVideo.write(prev_image_original if prev_image_original is not None else image_original)

            prev_image_original = image_original.copy()

outPutVideo.release()
cap.release()
