#!/usr/bin/env python3
from __future__ import print_function
 
import roslib
#roslib.load_manifest('my_package')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Int32,Bool

import numpy as np
####
import os



DATA_DIR = os.path.join(os.getcwd(), 'data')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
for dir in [DATA_DIR, MODELS_DIR]:
    if not os.path.exists(dir):
        os.mkdir(dir)

import tarfile
import urllib.request

# Download and extract model
MODEL_DATE = '20200711'
MODEL_NAME = 'my_model4'
#MODEL_TAR_FILENAME = MODEL_NAME + '.tar.gz'
#MODELS_DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/tf2/'
#MODEL_DOWNLOAD_LINK = MODELS_DOWNLOAD_BASE + MODEL_DATE + '/' + MODEL_TAR_FILENAME
#PATH_TO_MODEL_TAR = os.path.join(MODELS_DIR, MODEL_TAR_FILENAME)
PATH_TO_CKPT = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'checkpoint/'))
PATH_TO_CFG = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'pipeline.config'))


# Download labels file
LABEL_FILENAME = 'label_map.pbtxt'
#LABELS_DOWNLOAD_BASE = \
#    'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
PATH_TO_LABELS = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, LABEL_FILENAME))


# %%
# Load the model
# ~~~~~~~~~~~~~~
# Next we load the downloaded model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])


# %%
# Load label map data (for plotting)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Label maps correspond index numbers to category names, so that when our convolution network
# predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility
# functions, but anything that returns a dictionary mapping integers to appropriate string labels
# would be fine.
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

# %%
# Define the video stream


####
def callback1(data):
    if data.data ==1:
        global lowest_cx
        global lowest_cy
        lowest_cx=639
        lowest_cy=479  
 
def callback(data):
    try:
      global cv_image  
      #cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
      cv_image = bridge.imgmsg_to_cv2(data, "rgb8")
#     print(type(cv_image)) 
#     print(cv_image.ndim)
#     print(cv_image.shape)  

    except CvBridgeError as e:
      print(e)

   

    #print(a)
    image_np=cv_image
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    #image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()





    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
#          detections['detection_boxes'][0].numpy(),
#          (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
#          detections['detection_scores'][0].numpy(),
          detections['detection_boxes'][0].numpy(),
          (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
          detections['detection_scores'][0].numpy(),
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.2,
          agnostic_mode=False)

# Method 1
    boxes = detections['detection_boxes'].numpy()[0]
# get all boxes from an array
    max_boxes_to_draw = boxes.shape[0]
# get scores to get a threshold
    scores = detections['detection_scores'].numpy()[0]
# this is set as a default but feel free to adjust it to your needs
    min_score_thresh=.1
 # iterate over all objects found
    coordinates = []
   
    uv_pub1=rospy.Publisher("u_coordinates",Int32)
    uv_pub2=rospy.Publisher("v_coordinates",Int32)

    global lowest_cx
    global lowest_cy
    
    cx_list=[]
    cy_list=[]
    
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):

        if int(detections['detection_classes'].numpy()[0][i] + 1)==2 and scores[i] > min_score_thresh:
            class_id = int(detections['detection_classes'].numpy()[0][i] + 1)
            coordinates.append({
            "box": boxes[i],
            "class_name": category_index[class_id]["name"],
            "score": scores[i]
            })
            [ymin,xmin,ymax,xmax]=boxes[i]
            print(ymin)
            print(xmin)
            print(ymax)
            print(xmax)
            cx=(xmin+xmax)/2*640
            cy=(ymin+ymax)/2*480
            print('cx:',cx)
            print('cy:',cy)

            cx_list.append(cx)
            cy_list.append(cy)
#            if cx<lowest_cx:
#                lowest_cx=cx
#                lowest_cy=cy
#            else:
#                lowest_cx=lowest_cx
#                lowest_cy=lowest_cy

#            if cy<lowest_cy:
#                lowest_cy=cy
#            else:
#                lowest_cy=lowest_cy

#            print('lowest cx:',lowest_cx)
#            print('lowest cy:',lowest_cy) 
            u=round(cx_list[0])
            v=round(cy_list[0])
            print('First in cx list:',u)
            print('First in cy list:',v)              
            uv_pub1.publish(u)
            uv_pub2.publish(v)
            cv2.circle(image_np_with_detections,(u,v),10,(0,0,255),2)
#            cv2.circle(image_np_with_detections,(round(cx),round(cy)),10,(0,0,255),2)
#            cv2.circle(image_np_with_detections,(round(lowest_cx),round(lowest_cy)),25,(255,0,0),3)

    
    print(coordinates)




    # Display output
    #cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))
    image_np_with_detections_bgr = cv2.cvtColor(image_np_with_detections, cv2.COLOR_RGB2BGR)
    cv2.imshow('object detection',image_np_with_detections_bgr)
    cv2.waitKey(3)
    
def sub():
   image_sub = rospy.Subscriber("/Kinect/color/image_raw",Image,callback)
   rospy.spin()
 
def sub_reset():
   reset_recieved = rospy.Subscriber('reset_flag',Bool,callback1)

#if __name__=='__main__':
sub_reset()
lowest_cx=639
lowest_cy=479 
rospy.init_node('image_converter', anonymous=True)
bridge = CvBridge()
sub()
cv2.destroyAllWindows()
