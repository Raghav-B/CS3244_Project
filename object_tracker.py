import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # comment out line to enable tensorflow logging outputs
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # uncomment to use CPU instead of GPU

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
import time
import threading
from PIL import Image
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import random
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from video_reader import VideoReader

# Deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')


def main(_argv):
    # Hyperparameters
    max_cosine_distance = 0.4 # Used in deep sort
    nn_budget = None # Used in deep sort
    nms_max_overlap = 1.0
    gathering_thresh = 5
    
    # Initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1) # This encodes the data inside a bounding box into a vector
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget) # Calculate cosine distance metric
    tracker = Tracker(metric) # Initialize tracker
    print("Deep sort initialized")

    # Initialize DBSCAN model for clustering
    dbscan_model = DBSCAN(eps=150, min_samples=1)
    print("DBSCAN initialized")

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # Load object detection model
    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']
    print("Object detection model initialized")

    # Read in all class names from config
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)
    # Only allowed_classes will be drawn
    allowed_classes = ['person'] #list(class_names.values())

    # Display/Visual things
    video = None
    out = None
    if not FLAGS.output:
        #video = VideoReader(video_path) # Initialize reader for video
        video = cv2.VideoCapture(video_path)
    else:
        video = cv2.VideoCapture(video_path)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    display_yolo = False # Controls whether yolo bounding boxes are drawn
    display_centroids = False # Controls whether bounding box centroids are drawn
    display_groups = True
    video_written = False

    cv2.namedWindow("detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("detection", 1280, 720)

    # Main loop
    while True:
        _, frame = video.read()
        if frame is None:
            print('Video has ended, restarting video...')
            
            if FLAGS.output:
                if not video_written:
                    print("Output video written!")
                    out.release()
                    video_written = True
            
            video.reset()
            tracker.reset_tracks()
            _, frame = video.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Reformatting frame read from video
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # Convert frame image data to tensorflow input matrix and perform prediction
        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        # On predictions, run NMS to clean duplicate bounding boxes outputted by the model
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # Convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # Get usable format of bounding boxes
        original_h, original_w, _ = frame.shape
        bboxes, bboxes_xyxy = utils.format_boxes(bboxes, original_h, original_w)

        # Drawing YOLO detected bounding boxes
        # if display_yolo:
        #     for j in range(0, len(bboxes_xyxy)):
        #         if classes[j] != 0:
        #             continue
        #         box = bboxes_xyxy[j]
        #         cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)

        # Loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # Delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # Get list of centroids for YOLO detections
        centroids = [utils.get_centroid(bboxes[i]) for i in range(0, len(bboxes))]

        # Initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]    

        # Detect and draw clusters of people too close together
        cluster_bboxes = []
        cluster_sizes = []
        if len(centroids) != 0:
            cluster_assignments = dbscan_model.fit_predict(centroids)
            clusters = np.unique(cluster_assignments)

            for cluster in clusters:
                # Get array of centroids detected to be under current cluster
                row_ix = np.where(cluster_assignments == cluster) 

                color = colors[random.randint(0,10) % len(colors)]
                color = [j * 255 for j in color]

                point_arr = []
                for i in row_ix[0]:
                    point_arr.append(bboxes_xyxy[i][0:2])
                    point_arr.append(bboxes_xyxy[i][2:4])

                    if display_centroids:
                        frame = cv2.circle(frame, (centroids[i][0], centroids[i][1]), 20, color, -1)

                # Get bounding rectangle that covers group of people
                x,y,w,h = cv2.boundingRect(np.array(point_arr))
                
                # Skip all bounding rectangles that have height or width of 1
                if w != 1 and h != 1:                  
                    cluster_bboxes.append([x,y,w,h]) # Store bounding box of cluster
                    cluster_sizes.append(len(point_arr)//2) # Store number of people in current cluster

        # Encode cluster bboxes and feed to tracker
        features = encoder(frame, cluster_bboxes) # Get feature vectors of all bounding boxes
        detections = [Detection(bbox, num_people, feature) for bbox, num_people, feature in zip(cluster_bboxes, cluster_sizes, features)]
        
        # Call the Deep sort tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        for track in tracker.tracks:
            # If track is not confirmed or if it has been 2 or more frames since the track
            # was not found, we do not draw this track. The deletion of this track will
            # be handled automatically by the tracker.update() function I believe
            if not track.is_confirmed() or track.time_since_update > 1 or \
                    track.num_people <= gathering_thresh:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            
            # Draw detection box on screen
            blank_frame = np.zeros(frame.shape, np.uint8)
            cv2.rectangle(blank_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,0,0), cv2.FILLED)
            frame = cv2.addWeighted(frame, 1.0, blank_frame, 0.4, 1)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,0,0), 2)
            cv2.putText(frame, f"Size: {track.num_people}", 
                    tuple(track.get_centroid()), 0, 2, (255,255,255), 2)

            # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            #cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            
        
            # TODO If track age is greater than a certain number of frames, we issue an alert! Omg I'm done yay.

            # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        # Calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("detection", result)
        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)

        key_press = cv2.waitKey(1) & 0xFF
        if key_press == ord('q') or key_press == 27: # ESC key
            break
        elif key_press == ord('y'):
            display_yolo = not display_yolo
        elif key_press == ord('c'):
            display_centroids = not display_centroids

    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
