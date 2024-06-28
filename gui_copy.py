import sys
import time

from numpy import block
from PySide6.QtCore import QSize
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QTextEdit, QTabWidget, QFileDialog, QToolButton, QFileDialog, QLineEdit, QHBoxLayout, QMessageBox, QGraphicsBlurEffect, QGraphicsOpacityEffect, QLabel, QTableWidget, QTableWidgetItem
from PySide6.QtGui import QFont, QTextBlockFormat, QTextCursor, QImage, QTextDocument, QTextImageFormat
from PySide6.QtCore import Qt, QPropertyAnimation, Slot, QEvent, QObject, Signal, QThread
import cv2
import numpy as np
import threading
from typing import List
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
import os
from sympy import N
from ultralytics import YOLO
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from cProfile import label
from tabnanny import verbose
from tqdm.notebook import tqdm
from numpy import argmax
import pandas as pd
from datetime import timedelta
from collections import defaultdict, OrderedDict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from scipy import interpolate
from scipy.spatial import distance as dist
import pickle
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import directed_hausdorff
from supervision.tools.line_counter import LineCounter

from sklearn.manifold import TSNE
from scipy.interpolate import interp1d

import shutil

from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()

######################
# Set Home Directory #
######################

import os
HOME = os.getcwd()
print(HOME)

######################
# Import yolox modek #
######################

# %cd {HOME}
# !git clone https://github.com/ifzhang/ByteTrack.git
# %cd {HOME}/ByteTrack

# # workaround related to https://github.com/roboflow/notebooks/issues/80
# !sed -i 's/onnx==1.8.1/onnx==1.9.0/g' requirements.txt

# !pip3 install -q -r requirements.txt
# !python3 setup.py -q develop
# !pip install -q cython_bbox
# !pip install -q onemetric
# # workaround related to https://github.com/roboflow/notebooks/issues/112 and https://github.com/roboflow/notebooks/issues/106
# !pip install -q loguru lap thop

from IPython import display
display.clear_output()


import sys
sys.path.append(f"{HOME}/ByteTrack")


import yolox
print("yolox.__version__:", yolox.__version__)

##########################
# Import ByteTrack model #
##########################

from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.10
    track_buffer: int = 500
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


######################
# Import Supervision #
######################
    
# !pip install supervision==0.1.0


from IPython import display
display.clear_output()


import supervision
print("supervision.__version__:", supervision.__version__)

from supervision.draw.color import ColorPalette
from supervision.draw.color import Color
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.geometry.dataclasses import Point, Vector

####################### Setups #######################

from typing import List

import numpy as np

from formatter import NullFormatter
import cv2
import matplotlib.pyplot as plt

# Load the siamese network


import torch
import torch.nn as nn
import torch.optim as optim
# Import F
import torch.nn.functional as F
from torchvision import transforms

target_shape = (256, 256)
    
import torch
import torch.nn as nn
import torchvision.models as models
# Import PIL
from PIL import Image

base_model = models.resnet50(pretrained=True)

# Remove the classification head
modules = list(base_model.children())[:-1]

# Freeze the layers until conv5_block1_out
trainable = False
for name, child in base_model.named_children():
    if name == "layer4":
        trainable = True
    for param in child.parameters():
        param.requires_grad = trainable

# Add output layers
fc1 = nn.Sequential(
    nn.Flatten(),
    nn.Linear(2048, 1024),
    nn.ReLU(),
    nn.BatchNorm1d(1024),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Linear(512, 256),
)

# Create the embedding network
embedding_network = nn.Sequential(*modules, fc1)

class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.embedding_network = embedding_network

    def forward_once(self, x):
        x = self.embedding_network(x)
        return x
    
    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)

        return out1, out2

    
siamese = SiameseNet().cuda()
siamese_pedestrians = SiameseNet().cuda()


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
    
criterion = ContrastiveLoss()
# Print the number of parameters in the model
print(f"Number of parameters in the model: {sum(p.numel() for p in siamese.parameters())}")
optimizer = optim.Adam(siamese.parameters(), lr = 0.0005 )

# Load the model
siamese.load_state_dict(torch.load('siamese_model_contrastive_more_epochs.pth'))
siamese_pedestrians.load_state_dict(torch.load('siamese_model_contrastive_pedestrians_1.pth'))



transform = transforms.Compose(
    [
        transforms.Resize((target_shape[0], target_shape[1])),
        transforms.ToTensor(),
    ]
)

def determine_last_track_id(detection, track_history, frame, frame_nr, frame_history, frame_eps = 120*30, eps = 100, siam_thresh = 0.5, siamese=siamese):
    ''' If the track was interrupted, the object will be assigned a new track_id.
        This function returns the last track_id of the object.
    '''

    while len(frame_history) > frame_eps:
        frame_history.popitem(last=False)


    last_track_id = None
    track_id = detection[-1]
    x1, y1, x2, y2 = detection[0]
    x, y = float(x1 + x2) / 2, float(y1 + y2) / 2
    # x, y = detection[0]

    # If the object is not new, we need to determine the last track_id
    if track_id in track_history:
        return None


    
    # Verify if the object is towards the center of the frame
    padding_width = 0.15 * frame.shape[1] # 10% of the frame width
    padding_height = 0.15 * frame.shape[0] # 10% of the frame height


    # If the detection is outside the padding area, we consider it as a new object and thus we do not need to determine the last track_id
    if x < padding_width or x > frame.shape[1] - padding_width or y < padding_height or y > frame.shape[0] - padding_height:
        return None

    # Proximity analysis
    min_distance = 1000000
    last_box = None
    last_frame_nr = None
    # Order track_history by the timestamp of the last position in every track
    for existing_track, positions_frames in list(track_history.items()):
        if positions_frames:
            positions = [position[0] for position in positions_frames]
            frame_nrs = [position[1] for position in positions_frames]

            if len(positions) == 0:
                continue

            # Verif
            last_position = positions[-1]
            last_x1, last_y1, last_x2, last_y2 = last_position
            last_x = float(last_x1 + last_x2) / 2
            last_y = float(last_y1 + last_y2) / 2
            # print(last_x1, last_y1)
            # print(positions)
            
            # Verify if the frame number is close to the last frame number of the track
            if frame_nr < frame_nrs[-1] or frame_nr - frame_nrs[-1] > frame_eps:
                continue

            distance = np.sqrt((x - last_x)**2 + (y - last_y)**2)

            if distance > eps:
                continue
             
            # Verify the angle between the two trajectories
            # Get the last two positions of the current track and the last two positions of the lost track
            if len(positions) > 1:
                if min_distance == -1 or distance < min_distance:
                    min_distance = distance
                    last_track_id = existing_track
                    last_box = [last_x1, last_y1, last_x2, last_y2]
                    last_frame_nr = frame_nrs[-1]

    # Verify if the two objects are the same
    if last_box is None:
        return None
    frame_obj1 = frame
    frame_obj2 = frame_history[last_frame_nr]
    obj1 = frame_obj1[int(y1):int(y2), int(x1):int(x2)]
    # Transfor to PIL image
    obj1 = Image.fromarray(obj1)
    # Resize the image
    obj1 = transform(obj1)
    obj2 = frame_obj2[int(last_box[1]):int(last_box[3]), int(last_box[0]):int(last_box[2])]
    # Transfor to PIL image
    obj2 = Image.fromarray(obj2)
    obj2 = transform(obj2)
    obj1 = obj1.unsqueeze(0).cuda()
    obj2 = obj2.unsqueeze(0).cuda()


    siamese.eval()

    output1, output2 = siamese(obj1, obj2)

    euclidean_distance = F.pairwise_distance(output1, output2)
    if euclidean_distance <= siam_thresh:
        return last_track_id
    else:
        return None

    return last_track_id

class Polygon:
    def __init__(self, points):
        self.points = points
        # edges = list of vectors, 1-2 2-3 3-4 ... (n-1)-n n-1
        self.edges = [Vector(start=points[i], end=points[i+1]) for i in range(len(points)-1)] + [Vector(start=points[-1], end=points[0])]

    def is_in(self, point: Point) -> bool:
        """
        Check if a point is inside the polygon.

        :param point: Point : The point to check.
        :return: bool : True if the point is inside the polygon, False otherwise.
        """
        # see how many edges the vector from the point to the right crosses
        # if the number is odd, the point is inside the polygon
        # if the number is even, the point is outside the polygon
        # if the number is 0, the point is on the edge of the polygon
        counter = 0
        for edge in self.edges:
            if point.y > min(edge.start.y, edge.end.y):
                if point.y <= max(edge.start.y, edge.end.y):
                    if point.x <= max(edge.start.x, edge.end.x):
                        if edge.start.y != edge.end.y:
                            x_intersection = (point.y - edge.start.y) * (edge.end.x - edge.start.x) / (edge.end.y - edge.start.y) + edge.start.x
                            if edge.start.x == edge.end.x or point.x <= x_intersection:
                                counter += 1
        return counter % 2 != 0
    
class LineCounter(LineCounter):
    def update(self, detections: Detections):
        """
        Update the in_count and out_count for the detections that cross the line.

        :param detections: Detections : The detections for which to update the counts.
        """
        for xyxy, confidence, class_id, tracker_id in detections:
            # If there is no tracker_id, we skip the detection
            if tracker_id is None:
                continue

            # See how many points are on each side of the line
            x1, y1, x2, y2 = xyxy
            # anchors = [
            #     Point(x=x1, y=y1),
            #     Point(x=x1, y=y2),
            #     Point(x=x2, y=y1),
            #     Point(x=x2, y=y2),
            # ]

            # Workaround for the fact that the bboxes are too big

            '''
                Introduce a padding so the bbox will be the one in the interior of the original one, like so:
                |-----------------------|
                |  \                  / |
                |   \---------------/   |
                |   |Bbox w padding |   |
                |   /---------------\   |
                | /                  \  |
                |-----------------------|
            
            '''
            # Preduce the box by 40% of its original size
            percentange = 0.3
            anchors = [
                Point(x=x1 + (x2 - x1) * percentange, y=y1 + (y2 - y1) * percentange),
                Point(x=x1 + (x2 - x1) * percentange, y=y2 - (y2 - y1) * percentange),
                Point(x=x2 - (x2 - x1) * percentange, y=y1 + (y2 - y1) * percentange),
                Point(x=x2 - (x2 - x1) * percentange, y=y2 - (y2 - y1) * percentange),
            ]


            # Bool list. The truth value indicates the side of the line the point is on.
            triggers = [self.vector.is_in(point=anchor) for anchor in anchors]

            # detection is partially in and partially out
            if len(set(triggers)) == 2:
                continue

            tracker_state = triggers[0]
            # handle new detection
            if tracker_id not in self.tracker_state:
                self.tracker_state[tracker_id] = tracker_state
                continue

            # handle detection on the same side of the line
            if self.tracker_state.get(tracker_id) == tracker_state:
                continue

            self.tracker_state[tracker_id] = tracker_state
            if tracker_state:
                self.in_count += 1
            else:
                self.out_count += 1

# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: Detections,
    tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,)), np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)
    id_track_corelation = dict()

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id
            id_track_corelation[tracker_ids[detection_index]] = tracks[tracker_index]

    return tracker_ids, id_track_corelation


##################
# YoloX settings #
##################
from ultralytics import YOLO

MODEL = "yolov8x.pt"


model = YOLO(MODEL)
model.fuse()


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class UserEvent(QEvent):
    def __init__(self):
        super().__init__(QEvent.Type(QEvent.User))

class Run(QObject):
    finished = Signal()
    displayWarning = Signal()
    def __init__(self, filename, output, zones):
        super().__init__()
        self.filename = filename
        self.output = output
        self.zones = zones
        self.paths_constants()
        generator = get_video_frames_generator(self.SOURCE_VIDEO_PATH)
        iterator = iter(generator)
        frame = next(iterator)
        print(frame.shape)
        self.polygons = self.select_polygons(frame)
        self.lines = self.select_lines(frame)
        self.w1 = 0.20
        self.w2 = 0.33
        self.w3 = 0.47

        # Set up the line counters
        self.line_counters = []
        for i, line in enumerate(self.lines):
            self.line_counters.append(LineCounter(start=line[0], end=line[1]))
        



    def paths_constants(self):
        
        

        ###################### Paths #######################

        self.HOME = os.getcwd()
        print(self.HOME)

        self.SOURCE_VIDEO_PATH = self.filename

        self.TARGET_VIDEO_PATH = f"{self.HOME}/{self.output}.mp4"

        print(f"SOURCE_VIDEO_PATH: {self.SOURCE_VIDEO_PATH}")
        print(f"TARGET_VIDEO_PATH: {self.TARGET_VIDEO_PATH}")

        VideoInfo.from_video_path(self.SOURCE_VIDEO_PATH)

        self.MODEL = "yolov8x.pt"


        self.model = YOLO(self.MODEL)
        self.model.fuse()

        ###################### Class constants #######################

        # dict maping class_id to class_name
        self.CLASS_NAMES_DICT = self.model.model.names
        # class_ids of interest - person, car, motorcycle, bus and truck
        self.CLASS_ID = [0, 2, 3, 5, 7]
        self.CLASS_ID_PEOPLE = [0]
        self.CLASS_ID_VEHICLE = [2, 3, 5, 7]
        self.CLASS_ID_TRAFFIC_LIGHT = [9]

        
        print("Finished, signal emitted")

    def select_polygons(self, image):
        """
        Allows the user to select polygons on an image using mouse clicks until 'Enter' is pressed.
        
        :param image: The image on which polygons will be selected.
        """
        # Initialize global variables
        points = []  # To store the points for the current polygon
        polygons = []  # To store all polygons

        resized_img = cv2.resize(image, (1000, 800))

        # Calculate scale factors
        originalHeight, originalWidth = image.shape[:2]
        scaleFactorX = originalWidth / 1000
        scaleFactorY = originalHeight / 800


        def click_event(event, x, y, flags, param):
            nonlocal points, image
            if event == cv2.EVENT_LBUTTONDOWN:  # Left button click
                # scale the points back to the original image size
                x_scaled = int(x * scaleFactorX)
                y_scaled = int(y * scaleFactorY)
                points.append(Point(x_scaled, y_scaled))
                cv2.circle(resized_img, (x, y), 5, (0, 0, 255), -1)  # Draw the dot
                if len(points) > 1:
                    cv2.line(resized_img, 
                        (int(points[-2].x // scaleFactorX), int(points[-2].y // scaleFactorY)),
                        (int(points[-1].x // scaleFactorX), int(points[-1].y // scaleFactorY)), 
                        (255, 0, 0), 2)
                cv2.imshow("image", resized_img)  # Show the image with the new dot/line
                
            elif event == cv2.EVENT_RBUTTONDOWN:  # Right button to finish the polygon
                if len(points) > 2:  # Need at least 3 points to form a polygon
                    cv2.line(resized_img, 
                        (int(points[-1].x // scaleFactorX), int(points[-1].y // scaleFactorY)),
                        (int(points[0].x // scaleFactorX), int(points[0].y // scaleFactorY)), 
                        (255, 0, 0), 2)
                    polygons.append(points.copy())  # Store the completed polygon
                    points.clear()  # Reset points for the next polygon
                cv2.imshow("image", resized_img)

        cv2.namedWindow("image")
        cv2.setMouseCallback("image", click_event)
        cv2.imshow("image", resized_img)

        while True:  # Loop to keep the window open until 'Enter' is pressed
            key = cv2.waitKey(0) & 0xFF
            if key == 13:  # ASCII code for Enter key
                break

        cv2.destroyAllWindows()

        for i, polygon in enumerate(polygons):
            print(f"Polygon {i+1}: {[f'({p.x}, {p.y})' for p in polygon]}")

        return polygons
    
    def select_lines(self, img):
        # Initialize global variables
        points = []  # To store the points where you click
        lines = []

        resized_img = cv2.resize(img, (1000, 800))

        # Calculate scale factors
        originalHeight, originalWidth = img.shape[:2]
        scaleFactorX = originalWidth / 1000
        scaleFactorY = originalHeight / 800

        # Callback function for mouse events
        def click_event(event, x, y, flags, param):
            nonlocal points, img, scaleFactorX, scaleFactorY
            if event == cv2.EVENT_LBUTTONDOWN:  # Left button click
                if len(points) < 2:  # Ensure we only have 2 points
                    # Adjust x, y back to original image scale
                    origX = int(x * scaleFactorX)
                    origY = int(y * scaleFactorY)
                    points.append((origX, origY))
                    cv2.circle(resized_img, (x, y), 5, (0, 0, 255), -1)  # Draw the dot on resized image
                    if len(points) == 2:
                        lines.append((points[0], points[1]))
                        cv2.line(resized_img, 
                            (int(points[0][0] // scaleFactorX), int(points[0][1] // scaleFactorY)),
                            (int(points[1][0] // scaleFactorX), int(points[1][1] // scaleFactorY)), 
                            (255, 0, 0), 2)  # Draw the line on resized image
                        print(f"Point 1: {points[0]}, Point 2: {points[1]}")  # Print coordinates of original points
                        cv2.imshow("image", resized_img)  # Show the image with the line
                        
            if len(points) == 2:  # Reset after 2 points for new line drawing
                points.clear()


        cv2.namedWindow("image")
        cv2.setMouseCallback("image", click_event)

        cv2.imshow("image", resized_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        lines_return = []
        for i, line in enumerate(lines):
            # line_start = Point(lines[0][0][0], lines[0][0][1])
            # line_end = Point(lines[0][1][0], lines[0][1][1])
            line_start = Point(line[0][0], line[0][1])
            line_end = Point(line[1][0], line[1][1])
            lines_return.append((line_start, line_end))
            print(f"Line {i+1}: {line_start}, {line_end}")

        return lines_return
    
    def reduce_trajectory(self, trajectory, num_points):
        """
        Reduce the number of points in a trajectory, using interpolation, to a given number of points.
        Each segment between points will have approximately the same length.
        """
        if not isinstance(trajectory, np.ndarray) or trajectory.shape[1] != 2:
            raise ValueError("Trajectory must be a numpy array of shape (n, 2)")

        # Calculate cumulative distances along the trajectory
        distances = np.cumsum(np.sqrt(np.sum(np.diff(trajectory, axis=0)**2, axis=1)))
        distances = np.insert(distances, 0, 0)  # Insert the starting point distance (0)

        # Create an interpolation function based on the distances and trajectory coordinates
        interp_func_x = interp1d(distances, trajectory[:, 0], kind='linear')
        interp_func_y = interp1d(distances, trajectory[:, 1], kind='linear')

        # Evenly spaced distance values for interpolation
        target_distances = np.linspace(0, distances[-1], num_points)

        # Generate the new trajectory
        new_trajectory_x = interp_func_x(target_distances)
        new_trajectory_y = interp_func_y(target_distances)
        new_trajectory = np.column_stack((new_trajectory_x, new_trajectory_y))

        return new_trajectory
    
    def filter_length(self, trajectory, min_length=50):
        """
            Filter trajectories by length.
        """
        length = 0
        for i in range(1, len(trajectory)):
            length += dist.euclidean(trajectory[i], trajectory[i-1])

        return length >= min_length
        
        
    def count_trajectories(self, traffic_zones, detections, previous_placements):
        # Define traffic zones trajectories
        # 1-2, 1-3, 1-4, 2-3, 2-4, 3-4, etc
        # Transform traffic zones into polygons
        polygons = [Polygon(polygon) for polygon in traffic_zones]
        nr_zones = len(polygons)
        counts = dict()

        # See in which zone the detection is
        # For every detection verify all the polygons and see if it is in one of them
        detection_in_zone = dict()
        for xyxy, confidence, class_id, tracker_id in detections:
            # If there is no tracker_id, we skip the detection
            if tracker_id is None:
                continue

            x1, y1, x2, y2 = xyxy
            # anchors = [
            #     Point(x=x1, y=y1),
            #     Point(x=x1, y=y2),
            #     Point(x=x2, y=y1),
            #     Point(x=x2, y=y2),
            # ]

            # Workaround for the fact that the bboxes are too big

            '''
                Introduce a padding so the bbox will be the one in the interior of the original one, like so:
                |-----------------------|
                |  \                  / |
                |   \---------------/   |
                |   |Bbox w padding |   |
                |   /---------------\   |
                | /                  \  |
                |-----------------------|
            
            '''
            # Preduce the box by 40% of its original size
            percentange = 0.3
            anchors = [
                Point(x=x1 + (x2 - x1) * percentange, y=y1 + (y2 - y1) * percentange),
                Point(x=x1 + (x2 - x1) * percentange, y=y2 - (y2 - y1) * percentange),
                Point(x=x2 - (x2 - x1) * percentange, y=y1 + (y2 - y1) * percentange),
                Point(x=x2 - (x2 - x1) * percentange, y=y2 - (y2 - y1) * percentange),
            ]

            for i, polygon in enumerate(polygons):
                ok = True
                # For every side of the polygon chech if the 4 points are inside the polygon
                for anchor in anchors:
                    if not polygon.is_in(anchor):
                        ok = False
                        break
                if ok:
                    # If the detection is in the polygon, increment the counter
                    detection_in_zone[tracker_id] = i


            # Check if the detection changed the zone
            # PS: We ignore if the detection is in the same zone or the detection system didn't assign it to a zone
            if tracker_id in previous_placements:
                if tracker_id in detection_in_zone and detection_in_zone[tracker_id] != previous_placements.get(tracker_id, -1) \
                and (detection_in_zone[tracker_id] is not None or i >= 0):
                    if (previous_placements[tracker_id], detection_in_zone[tracker_id]) not in counts:
                        counts[(previous_placements[tracker_id], detection_in_zone[tracker_id])] = 1
                    elif previous_placements.get(tracker_id, -1) >= 0:
                        counts[(previous_placements[tracker_id], detection_in_zone[tracker_id])] += 1


        return counts, detection_in_zone, counts
    
    def annotate_with_counts(self, frame, time, counts, font_scale=1, thickness=2):
        for i, transitions in enumerate(counts.keys()):
            if counts[transitions] > 0:
                cv2.putText(frame, f"{transitions[0]+1} -> {transitions[1]+1}: {counts[transitions]}", (10, i * 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)

    def tsne_trajectory_corelation(self, trajectories, trajectories_tsne_first):
        """
            return a dictionary of the original trajectories and their t-SNE points.
        """
        tsne_dict = {}
        for i, traj in enumerate(trajectories):
            tsne_dict[tuple(traj.flatten())] = trajectories_tsne_first[i]
        return tsne_dict
    
    def average_trajectory(self, trajectories):
        """
            Average a set of trajectories.
        """
        # Get the number of trajectories and the number of points in each trajectory
        num_trajectories, num_points, _ = trajectories.shape
        # Create an array to store the average trajectory
        average = np.zeros((num_points, 2))
        # Average the trajectories
        for i in range(num_points):
            # Get the x and y coordinates of the points
            x = trajectories[:, i, 0]
            y = trajectories[:, i, 1]
            # Calculate the average
            average[i, 0] = np.mean(x)
            average[i, 1] = np.mean(y)
        # Smooth the trajectory
        average[:, 0] = gaussian_filter1d(average[:, 0], sigma=2)
        average[:, 1] = gaussian_filter1d(average[:, 1], sigma=2)
        return average
    
    def distance_similarity(self, traj1, traj2):
        """
            Calculate the Hausdorff distance between two trajectories.
        """
        # Calculate the directed Hausdorff distance
        distance1 = directed_hausdorff(traj1, traj2)[0]
        distance2 = directed_hausdorff(traj2, traj1)[0]
        return min(distance1, distance2)
    
    def angular_similarity(self, traj1, traj2):
        """
            Calculate the angular similarity between two trajectories.
        """

        sum1 = 0
        for i in range(1, len(traj1)):
            sum1 += np.linalg.norm(traj1[i] - traj1[i-1])
        sum2 = 0
        for i in range(1, len(traj2)):
            sum2 += np.linalg.norm(traj2[i] - traj2[i-1])
        if sum1 > sum2:
            # Swap the trajectories
            traj1, traj2 = traj2, traj1

        # Find the closest point of traj2 to the starting point of traj1
        min_distance = np.inf
        point1 = None
        for point in traj2:
            distance = np.linalg.norm(traj1[0] - point)
            if distance < min_distance:
                min_distance = distance
                point1 = point

        # Find the closest point of traj2 to the ending point of traj1
        min_distance = np.inf
        point2 = None
        for point in traj2:
            distance = np.linalg.norm(traj1[-1] - point)
            if distance < min_distance:
                min_distance = distance
                point2 = point

        # The two vectors are determined by the starting and ending points of the trajectories and the closest points of traj2
        vector1 = np.array(point1) - np.array(traj1[0])
        vector2 = np.array(point2) - np.array(traj1[-1])



        # Calculate the angle in degrees between the two vectors
        angle = np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))
        return np.degrees(angle)
    
    def rear_distance_similarity(self, traj1, traj2):
        """
            Calculate the rear distance similarity between two trajectories.
        """
        # Calculate the distance between the last point of traj1 and the first point of traj2
        distance = np.linalg.norm(traj1[-1] - traj2[0]) 
        if distance >= self.distance_similarity(traj1, traj2):
            return self.angular_similarity(traj1, traj2) / 360 * (distance - self.distance_similarity(traj1, traj2))
        if distance < self.distance_similarity(traj1, traj2) and self.angular_similarity(traj1, traj2) <= 15:
            return self.angular_similarity(traj1, traj2) / 360 * (self.distance_similarity(traj1, traj2) - distance)
        if distance < self.distance_similarity(traj1, traj2) and self.angular_similarity(traj1, traj2) > 15:
            return 0
    
    def predict(self):
        print("Setting up byte tracker instance")


        # create BYTETracker instance
        # Delete the existing tracker
        byte_tracker = BYTETracker(BYTETrackerArgs())
        # create VideoInfo instance
        video_info = VideoInfo.from_video_path(self.SOURCE_VIDEO_PATH)
        # create frame generator
        generator = get_video_frames_generator(self.SOURCE_VIDEO_PATH)
        box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=0.5)

        # Reset tracker


        # Timestamps for each frame
        cap = cv2.VideoCapture(self.SOURCE_VIDEO_PATH)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"fps: {fps}")
        timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
        calc_timestamps = [0.0]
        while(cap.isOpened()):
            frame_exists, curr_frame = cap.read()
            if frame_exists:
                timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
                calc_timestamps.append(calc_timestamps[-1] + 1000/fps)
            else:
                break
        cap.release()
        frame_time_stamp = dict()
        for i, (ts, cts) in enumerate(zip(timestamps, calc_timestamps)):
            if i not in frame_time_stamp:
                frame_time_stamp[i] = (timedelta(milliseconds=ts), ts)

        # open target video file
        detection_position = dict()
        transition_counts = dict()
        vehicle_counts = dict()
        pedestrian_counts = dict()
        track_history = defaultdict(lambda: [])
        track_history_pedestrians = defaultdict(lambda: [])
        # Frame history, a list of images
        frame_history = OrderedDict()
        with VideoSink(self.TARGET_VIDEO_PATH, video_info) as sink:
            # loop over video frames
            # for frame_nr, frame in enumerate(tqdm(generator, total=video_info.total_frames), start=1):        # model prediction on single frame and conversion to supervision Detections
            for frame_nr, frame in enumerate(generator):
                if frame_nr % 300 == 0:
                    print(f"frame_nr: {frame_nr} out of {video_info.total_frames}")
                # print(f"frame_nr: {frame_nr}")
                results = model(frame, verbose=False)
                detections = Detections(
                    xyxy=results[0].boxes.xyxy.cpu().numpy(),
                    confidence=results[0].boxes.conf.cpu().numpy(),
                    class_id=results[0].boxes.cls.cpu().numpy().astype(int)
                )
                # detections_people = Detections(
                #     xyxy=results[0].boxes.xyxy.cpu().numpy(),
                #     confidence=results[0].boxes.conf.cpu().numpy(),
                #     class_id=results[0].boxes.cls.cpu().numpy().astype(int)
                # )
                # detections_vehicles = Detections(
                #     xyxy=results[0].boxes.xyxy.cpu().numpy(),
                #     confidence=results[0].boxes.conf.cpu().numpy(),
                #     class_id=results[0].boxes.cls.cpu().numpy().astype(int)
                # )
                # filtering out detections with unwanted classes
                # mask_people = np.array([class_id in self.CLASS_ID_PEOPLE for class_id in detections_people.class_id], dtype=bool)
                # mask_vehicle = np.array([class_id in self.CLASS_ID_VEHICLE for class_id in detections_vehicles.class_id], dtype=bool)
                # detections_people.filter(mask=mask_people, inplace=True)
                # detections_vehicles.filter(mask=mask_vehicle, inplace=True)
                mask = np.array([class_id in self.CLASS_ID for class_id in detections.class_id], dtype=bool)
                detections.filter(mask=mask, inplace=True)

                tracks, tracked_stracks = byte_tracker.update(
                    output_results=detections2boxes(detections=detections),
                    img_info=frame.shape,
                    img_size=frame.shape
                )
                tracker_id, id_track_corelation = match_detections_with_tracks(detections=detections, tracks=tracks)
                # tracker_id_people, id_track_corelation_people = match_detections_with_tracks(detections=detections_people, tracks=tracks)
                # tracker_id_vehicles, id_track_corelation_vehicles = match_detections_with_tracks(detections=detections_vehicles, tracks=tracks)
                detections.tracker_id = np.array(tracker_id)
                # detections_people.tracker_id = np.array(tracker_id_people)
                # detections_vehicles.tracker_id = np.array(tracker_id_vehicles)


                # print('---')
                # print(len(detections))
                # print(len(detections_people))
                # print(len(detections_vehicles))

                # filtering out detections without trackers
                mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
                detections.filter(mask=mask, inplace=True)
                mask_people = np.array([class_id in self.CLASS_ID_PEOPLE for class_id in detections.class_id], dtype=bool)
                mask_vehicles = np.array([class_id in self.CLASS_ID_VEHICLE for class_id in detections.class_id], dtype=bool)
                detections_people = detections.filter(mask=mask_people, inplace=False)
                detections_vehicles = detections.filter(mask=mask_vehicles, inplace=False)

                # Update line counters
                for i in range(len(self.line_counters)):
                    self.line_counters[i].update(detections=detections_people)


                time = frame_time_stamp[frame_nr][1]

                # Verify if the object is new or if it was interrupted
                for i, (detection) in enumerate(detections_vehicles):
                
                    new_track_id = determine_last_track_id(detection, track_history, frame, frame_nr, frame_history, eps=50, siamese=siamese)
                    if new_track_id is not None:
                        if new_track_id  not in detections_vehicles.tracker_id and new_track_id not in detections.tracker_id and new_track_id not in detections_people.tracker_id:
                            print('---')
                            print("old: ", detection[-1])
                            print("new: ", new_track_id)         
                            # Find the track that was interrupted in the tracks list
                            for j, track in enumerate(tracked_stracks):
                                if track.track_id == detection[-1]:
                                    # print(new_track_id)
                                    print(tracked_stracks[j])
                                    tracked_stracks[j].track_id = new_track_id
                                    print(tracked_stracks[j])
                                    print(time / 1000)
                                    break
                            byte_tracker.update_strack(tracked_stracks)

                            detections_vehicles.tracker_id[i] = new_track_id

                for i, (detection) in enumerate(detections_people):
                
                    new_track_id = determine_last_track_id(detection, track_history_pedestrians, frame, frame_nr, frame_history, eps=50, siamese=siamese_pedestrians)
                    if new_track_id is not None:
                        if new_track_id  not in detections_vehicles.tracker_id and new_track_id not in detections.tracker_id and new_track_id not in detections_people.tracker_id:
                            print('---')
                            print("old: ", detection[-1])
                            print("new: ", new_track_id)         
                            # Find the track that was interrupted in the tracks list
                            for j, track in enumerate(tracked_stracks):
                                if track.track_id == detection[-1]:
                                    # print(new_track_id)
                                    print(tracked_stracks[j])
                                    tracked_stracks[j].track_id = new_track_id
                                    print(tracked_stracks[j])
                                    print(time / 1000)
                                    break
                            byte_tracker.update_strack(tracked_stracks)

                            detections_people.tracker_id[i] = new_track_id

                        
                        
                labels = [
                    f"#{tracker_id} {self.CLASS_NAMES_DICT[class_id]}"
                    for _, confidence, class_id, tracker_id
                    in detections_vehicles
                ]

                for i, (xyxy, confidence, class_id, track_id) in enumerate(detections_vehicles):
                    x1, y1, x2, y2 = xyxy.astype(int)
                    track = track_history[track_id] # this is getting the memory location of the list so it will be updated
                    # Filter
                    # if the point is too far from the last point, ignore it
                    if len(track) > 0 and np.linalg.norm(np.array(track[-1][0][:2]) - np.array([x1, y1])) > 100:
                        continue
                    track.append(((x1, y1, x2, y2), frame_nr))
                    frame_history[frame_nr] = frame

                # Do the same for pedestrians
                for i, (xyxy, confidence, class_id, track_id) in enumerate(detections_people):
                    x1, y1, x2, y2 = xyxy.astype(int)
                    track = track_history_pedestrians[track_id]
                    # Filter
                    # if the point is too far from the last point, ignore it
                    if len(track) > 0 and np.linalg.norm(np.array(track[-1][0][:2]) - np.array([x1, y1])) > 100:
                        continue
                    track.append(((x1, y1, x2, y2), frame_nr))


                traffic_zones, detection_in_zone, counts = self.count_trajectories(self.polygons, detections, detection_position)
                for _, confidence, class_id, tracker_id in detections:
                    if tracker_id is not None and detection_in_zone.get(tracker_id, -1) >= 0:
                        detection_position[tracker_id] = detection_in_zone.get(tracker_id, -1)
                for count in counts.keys():
                    if count not in transition_counts.keys():
                        transition_counts[count] = counts[count]
                    transition_counts[count] += counts[count]
                labels = [
                    f"#{tracker_id} {detection_in_zone.get(tracker_id, -1) + 1}" if tracker_id is not None else f"#{tracker_id}"
                    for _, confidence, class_id, tracker_id
                    in detections
                ]
                box_annotator.annotate(frame=frame, detections=detections, labels=labels)
                # # Write in the center of the polygon the number of the polygon
                # for i, polygon in enumerate(self.polygons):
                #     x = 0
                #     y = 0
                #     for point in polygon:
                #         x += point.x
                #         y += point.y
                #     x = int(x / len(polygon))
                #     y = int(y / len(polygon))
                #     cv2.putText(frame, f"{i+1}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # Annotate with counts
                self.annotate_with_counts(frame, 0, transition_counts)
                # # Draw the polygons
                # for i, polygon in enumerate(self.polygons):
                #     color = (255, 0, 0)
                #     if i == 0:
                #         color = (0, 255, 0)
                #     for j in range(len(polygon)):
                #         cv2.line(frame, (polygon[j].x, polygon[j].y), (polygon[(j+1) % len(polygon)].x, polygon[(j+1) % len(polygon)].y), color, 2)

                # Update counts
                for i, count in enumerate(transition_counts.keys()):
                    if f"from {count[0]+1} to {count[1]+1}" not in vehicle_counts:
                        vehicle_counts[f"from {count[0]+1} to {count[1]+1}"] = [(time / 1000, transition_counts[count])]
                    else:
                        vehicle_counts[f"from {count[0]+1} to {count[1]+1}"].append((time / 1000, transition_counts[count]))
                    
                for i, line_counter in enumerate(self.line_counters):
                    if (i, "in") not in pedestrian_counts:
                        pedestrian_counts[(i, "in")] = [(time / 1000, line_counter.in_count)]
                    else:
                        pedestrian_counts[(i, "in")].append((time / 1000, line_counter.in_count))


                    if (i, "out") not in pedestrian_counts:
                        pedestrian_counts[(i, "out")] = [(time / 1000, line_counter.out_count)]
                    else:
                        
                        pedestrian_counts[(i, "out")].append((time / 1000, line_counter.out_count))


                sink.write_frame(frame)

        # Save the results to csv
        df = pd.DataFrame(columns=["time", "id", "count"])
        for i, count in enumerate(vehicle_counts.keys()):
            for time, c in vehicle_counts[count]:
                new_df = pd.DataFrame([[time, count, c]], columns=["time", "id", "count"])
                df = pd.concat([df, new_df])
        df.to_csv("gui_utils/results/vehicle_counts.csv", index=False)

        df = pd.DataFrame(columns=["time", "id", "count"])
        for i, count in enumerate(pedestrian_counts.keys()):
            for time, c in pedestrian_counts[count]:
                new_df = pd.DataFrame([[time, count, c]], columns=["time", "id", "count"])
                df = pd.concat([df, new_df])
        df.to_csv("gui_utils/results/pedestrian_counts.csv", index=False)


        # Plot counts
        plt.figure(figsize=(10, 6))
        max_value = 0
        # Max time is the total time of the video
        max_time = video_info.total_frames / fps
        for i, count in enumerate(vehicle_counts.keys()):
            times, counts = zip(*vehicle_counts[count])
            times = (0,) + times
            counts = (0,) + counts
            # Add a point at the same y level just before the next point
            for j in range(1, len(times)):
                times = times[:j] + (times[j],) + times[j:]
                counts = counts[:j] + (counts[j-1],) + counts[j:]
            plt.plot(times, counts, label=f"{count}")
            # Find maximum y value for the line
            max_y = max(counts)
            xmax = times[counts.index(max_y)]
            ymax = max_y
            # Annotate the maximum point
            plt.annotate(max_y, xy=(xmax, ymax), xytext=(xmax, ymax+0.1),
                        arrowprops=dict(facecolor='black', arrowstyle='->', linewidth=1))
            if max_y > max_value:
                max_value = max_y
        plt.legend()
        x_label = "Timp (s)"
        y_label = "Numărul de pietoni"
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title("Numărul de pietoni ce trec dintr-o zonă în alta")
        plt.axis([0, max_time, 0, max_value + 1])
        plt.legend()
        plt.xticks(np.arange(0, max_time, 10))
        plt.yticks(np.arange(0, max_value + 1, 10))
        plt.savefig("gui_utils/results/vehicle_counts.png")

        plt.figure(figsize=(10, 6))
        max_value = 0
        for i, count in enumerate(pedestrian_counts.keys()):
            times, counts = zip(*pedestrian_counts[count])
            times = (0,) + times
            counts = (0,) + counts
            # Add a point at the same y level just before the next point
            for j in range(1, len(times)):
                times = times[:j] + (times[j],) + times[j:]
                counts = counts[:j] + (counts[j-1],) + counts[j:]
            plt.plot(times, counts, label=f"{count}")
            # Find maximum y value for the line
            max_y = max(counts)
            xmax = times[counts.index(max_y)]
            ymax = max_y
            # Annotate the maximum point
            plt.annotate(max_y, xy=(xmax, ymax), xytext=(xmax, ymax+0.1),
                        arrowprops=dict(facecolor='black', arrowstyle='->', linewidth=1))
            if max_y > max_value:
                max_value = max_y
        plt.legend()
        x_label = "Timp (s)"
        y_label = "Numărul de pietoni"
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title("Numărul de pietoni ce trec dintr-o zonă în alta")
        plt.axis([0, max_time, 0, max_value + 1])
        plt.legend()
        plt.xticks(np.arange(0, max_time, 10))
        plt.yticks(np.arange(0, max_value + 1, 10))
        plt.savefig("gui_utils/results/pedestrian_counts.png")

        # Save the data
        trajectories = []
        # Keep their temporal order
        for track_id in track_history.keys():
            track = track_history[track_id]
            if len(track) > 10: # Filter trajectories that are too short
                coordinates = []
                for position, frame_nr in track:
                    x1, y1, x2, y2 = position
                    x, y = (x1 + x2) / 2, (y1 + y2) / 2
                    coordinates.append([x, y])
                trajectories.append(np.array(coordinates))
        trajectories = [np.array(trajectory) for trajectory in trajectories]
        with open('trajectories_gui.pkl', 'wb') as f:
            pickle.dump(trajectories, f)

        # with open('trajectories2.pkl', 'rb') as f:
        #     trajectories = pickle.load(f)

        generator = get_video_frames_generator(self.SOURCE_VIDEO_PATH)
        frame = next(iter(generator))

        simplified_trajectories = [self.reduce_trajectory(trajectory, 200) for trajectory in trajectories if len(trajectory) > 50]

        length = 100

        # Filter the trajectories
        simplified_trajectories = [trajectory for trajectory in simplified_trajectories if self.filter_length(trajectory, min_length=length*2)]
        first_points = []
        last_points = []


        for trajectory in simplified_trajectories:
            i = 0
            l = 0
            while l < length:
                l += dist.euclidean(trajectory[i], trajectory[i+1])
                i += 1
            first_points.append(trajectory[:i])
            i = -1
            l = 0
            while l < length:
                l += dist.euclidean(trajectory[i], trajectory[i-1])
                i -= 1
            last_points.append(trajectory[i:])
            
        # Interpolate first and last points to have the same number of points
        first_points = [self.reduce_trajectory(trajectory, 50) for trajectory in first_points]
        last_points = [self.reduce_trajectory(trajectory, 50) for trajectory in last_points]

        np_trajectories = np.array(first_points)

        # Flatten each trajectory into a 1D array
        trajectories_flattened = np.array([traj.flatten() for traj in np_trajectories])

        # Perform t-SNE but keep a dictionary of the original trajectories and assign the tsne points to the original trajectories
        n_samples = len(trajectories_flattened)
        print(f"Number of samples: {n_samples}")

        perplexity = n_samples // 6
        tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity, n_iter=1000, learning_rate=300)
        trajectories_tsne_first = tsne.fit_transform(trajectories_flattened)

        tsne_dict = self.tsne_trajectory_corelation(first_points, trajectories_tsne_first)

        # Do the same for the last points
        np_trajectories = np.array(last_points)
        trajectories_flattened = np.array([traj.flatten() for traj in np_trajectories])

        trajectories_tsne_last = tsne.fit_transform(trajectories_flattened)

        # Standardize the points
        scaler = StandardScaler()
        trajectories_tsne_first_standardized = scaler.fit_transform(trajectories_tsne_first)
        trajectories_tsne_last_standardized = scaler.transform(trajectories_tsne_last)

        # Normalize the points
        trajectories_tsne_first_standardized = (trajectories_tsne_first_standardized - np.min(trajectories_tsne_first_standardized)) / (np.max(trajectories_tsne_first_standardized) - np.min(trajectories_tsne_first_standardized))
        trajectories_tsne_last_standardized = (trajectories_tsne_last_standardized - np.min(trajectories_tsne_last_standardized)) / (np.max(trajectories_tsne_last_standardized) - np.min(trajectories_tsne_last_standardized))

        cluster_model = SpectralClustering(n_clusters=self.zones)
        cluster_assignments = cluster_model.fit_predict(trajectories_tsne_first_standardized)
        cluster_assignments_last = cluster_model.fit_predict(trajectories_tsne_last_standardized)

        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.scatter(trajectories_tsne_first_standardized[:, 0], trajectories_tsne_first_standardized[:, 1], c=cluster_assignments)
        # Circle the clusters and label them
        for i in np.unique(cluster_assignments):
            cluster_points = trajectories_tsne_first_standardized[cluster_assignments == i]
            x = np.mean(cluster_points[:, 0])
            y = np.mean(cluster_points[:, 1])
            plt.text(x, y, str(i), fontsize=12, color='red')
            plt.gca().add_artist(plt.Circle((x, y), 0.1, color='red', fill=False))

        plt.savefig("gui_utils/results/first_points.png")

        # Plot the last points
        plt.figure(figsize=(10, 6))
        plt.scatter(trajectories_tsne_last_standardized[:, 0], trajectories_tsne_last_standardized[:, 1], c=cluster_assignments_last)
        # Circle the clusters and label them
        for i in np.unique(cluster_assignments_last):
            cluster_points = trajectories_tsne_last_standardized[cluster_assignments_last == i]
            x = np.mean(cluster_points[:, 0])
            y = np.mean(cluster_points[:, 1])
            plt.text(x, y, str(i), fontsize=12, color='red')
            plt.gca().add_artist(plt.Circle((x, y), 0.1, color='red', fill=False))

        plt.savefig("gui_utils/results/last_points.png")


        # Create a new figure
        plt.figure(figsize=(10, 6))

        # For each unique cluster identifier...
        for cluster_id in np.unique(cluster_assignments):
            # Get the indices of the points in this cluster
            cluster_indices = np.where(cluster_assignments == cluster_id)[0]
            
            # For each trajectory in this cluster...
            for i in cluster_indices:
                # Get the original trajectory
                traj = first_points[i]
                
                # Plot the trajectory
                plt.plot(traj[:, 0], traj[:, 1], color = plt.cm.tab20(cluster_id))

        plt.savefig("gui_utils/results/first_trajectories.png")


        first_clusters = cluster_assignments
        last_clusters = cluster_assignments_last

        all_trajectory_clusters = {}

        # Create 8 clusters depending on the first and last clusters
        for i, trajectory in enumerate(simplified_trajectories):
            # plt.figure(figsize=(10, 6))
            # plt.plot(trajectory[:, 0], trajectory[:, 1])
            # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # Get the first and last points
            first_10_points = trajectory[:10]
            last_10_points = trajectory[-10:]

            # print(first_10_points[0], last_10_points[0])

            cluster1 = first_clusters[i]
            cluster2 = last_clusters[i]
            

            if (cluster1, cluster2) in all_trajectory_clusters:
                all_trajectory_clusters[(cluster1, cluster2)].append(trajectory)
            else:
                all_trajectory_clusters[(cluster1, cluster2)] = [trajectory]
            

        print(len(all_trajectory_clusters.keys()))

        for cluster in all_trajectory_clusters.keys():
            print(f"Grupul {cluster} conține: {len(all_trajectory_clusters[cluster])} traiectorii")
            plt.figure(figsize=(10, 6))
            for trajectory in all_trajectory_clusters[cluster]:
                plt.plot(trajectory[:, 0], trajectory[:, 1])

            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # Plot the number of trajectories in the cluster
            plt.title(f"Grupul {cluster} conține: {len(all_trajectory_clusters[cluster])} traiectorii")
            plt.xlabel("Coordonata X")
            plt.ylabel("Coordonata Y")
            # Save the plot
            plt.savefig(f"gui_utils/results/clusters/cluster_{cluster}.png")

        average_trajectories = {}
        for cluster_key, trajectories in all_trajectory_clusters.items():
            # Stack trajectories for averaging
            trajectory_stack = np.stack(trajectories)
            average_trajectories[cluster_key] = self.average_trajectory(trajectory_stack)

        # Plotting the average trajectories
        plt.figure(figsize=(10, 6))
        for cluster_key, avg_traj in average_trajectories.items():
            plt.plot(avg_traj[:, 0], avg_traj[:, 1], label=f"Grupul {cluster_key}")
        plt.legend()
        plt.title("Traiectoria medie pentru fiecare grup")
        plt.xlabel("Coordonata X")
        plt.ylabel("Coordonata Y")
        plt.savefig("gui_utils/results/average_trajectories.png")

        # Plot on the frame
        plt.figure(figsize=(10, 6))
        for cluster_key, avg_traj in average_trajectories.items():
            plt.plot(avg_traj[:, 0], avg_traj[:, 1], label=f"Grupul {cluster_key}")
        plt.legend()
        plt.title("Traiectoriile medii pe imagine")
        plt.xlabel("Coordonata X")
        plt.ylabel("Coordonata Y")
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.savefig("gui_utils/results/average_trajectories_on_frame.png")

        # Plot every average trajectory separately
        plt.figure(figsize=(10, 6))
        for cluster_key, avg_traj in average_trajectories.items():
            plt.figure(figsize=(10, 6))
            # Plot the trajectory
            plt.plot(avg_traj[:, 0], avg_traj[:, 1], label=f"Grupul {cluster_key}", alpha=0.5)  # semi-transparent line
            # Add arrows to show the direction of the trajectory
            steps = len(avg_traj)  # Number of steps in the trajectory
            skip = max(1, steps // 20)  # Skip some points to avoid clutter
            for i in range(0, steps - 1, skip):
                # Determine start and end points for the arrows
                start_point = avg_traj[i]
                end_point = avg_traj[i + 1]
                plt.arrow(start_point[0], start_point[1], 
                        end_point[0] - start_point[0], end_point[1] - start_point[1],
                        shape='full', lw=0, color='red', length_includes_head=True, head_width=20, head_length=24, zorder=5)
            # Show image background if needed
            if 'frame' in locals():
                plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.legend()
            plt.title(f"Traiectoria medie pentru grupul {cluster_key}")
            plt.xlabel("Coordonata X")
            plt.ylabel("Coordonata Y")
            plt.savefig(f"gui_utils/results/average_trajectories/average_trajectory_{cluster_key}.png")

        for cluster in all_trajectory_clusters:
            trajectories = all_trajectory_clusters[cluster]
            # Find the least similar trajectories with the representative trajectory of the cluster
            representative_trajectory = average_trajectories[cluster]
            similarities = []
            for i, traj in enumerate(trajectories):
                similarities.append(self.w1 * self.distance_similarity(representative_trajectory, traj) + self.w2 * self.angular_similarity(representative_trajectory, traj) + self.w3 * self.rear_distance_similarity(representative_trajectory, traj))

            # Find the least similar trajectories
            least_similar = np.argsort(similarities)

            # If they are too similar, ignore them
            if similarities[least_similar[0]] < 45:
                continue

            # Plot the least similar trajectories
            plt.figure(figsize=(10, 6))
            plt.plot(trajectories[-1][:, 0], trajectories[-1][:, 1], color='red')
            plt.plot(representative_trajectory[:, 0], representative_trajectory[:, 1], color='blue')
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.savefig(f"gui_utils/results/least_similar/least_similar_{cluster}.png")


    def run(self):
        print("Running")
        # If there are images in thw gui_utils/results/average_trajectories folder, delete them
        if os.path.exists("gui_utils/results/average_trajectories"):
            shutil.rmtree("gui_utils/results/average_trajectories")
        os.makedirs("gui_utils/results/average_trajectories")
        # If there are images in thw gui_utils/results/least_similar folder, delete them
        if os.path.exists("gui_utils/results/least_similar"):
            shutil.rmtree("gui_utils/results/least_similar")
        os.makedirs("gui_utils/results/least_similar")
        # If there are images in thw gui_utils/results/clusters folder, delete them
        if os.path.exists("gui_utils/results/clusters"):
            shutil.rmtree("gui_utils/results/clusters")
        os.makedirs("gui_utils/results/clusters")
        self.predict()
        self.finished.emit()




class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.filename = None # Input file path
        self.output = None # Output file path
        self.zones = None # Number of zones
        self.setWindowTitle("Traffic Management System")
        self.setMinimumSize(QSize(1000, 700))
        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)
        layout = QVBoxLayout(centralWidget)

        self.setStyleSheet("""
            QMainWindow {
                border: 2px solid gray;
                background-color: gray;
                           
            }
        """)

        self.notebook = QTabWidget()
        self.tab = QWidget()
        self.notebook.addTab(self.tab, "Pagina principala")

        self.textEdit = QTextEdit(self.tab)
        self.textEdit.setReadOnly(True)
        self.textEdit.setFont(QFont('Arial', 14))
        self.tab1Layout = QVBoxLayout(self.tab)
        self.tab1Layout.addWidget(self.textEdit)


        self.display_paragraph("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed mi elit, cursus vitae pretium quis, pulvinar in tortor. Vestibulum ante lorem, luctus eget tortor sed, elementum faucibus orci. Quisque at consequat ex, a mattis elit. Mauris vel purus at ligula ornare semper. Nullam iaculis arcu quis congue vulputate. Cras in efficitur massa. Aliquam ut semper augue, sed egestas tortor. In venenatis venenatis mauris, ac cursus ligula porta nec. Duis semper tempus sodales. Cras at neque ac augue ultrices suscipit. Mauris sed tincidunt est. In egestas id diam sit amet ultrices. Praesent vel arcu ultricies, convallis sapien pellentesque, scelerisque ipsum. ", self.textEdit)
        self.display_paragraph("In lobortis sollicitudin cursus. Sed tincidunt elit eget dui sodales, at congue diam ultricies. Duis urna turpis, tincidunt vulputate finibus id, consectetur non nibh. Morbi ut elit cursus nibh elementum cursus sed quis libero. Donec interdum molestie velit sit amet pulvinar. Etiam massa ex, imperdiet quis urna id, iaculis ultrices eros. Vivamus lacinia, mauris vulputate lobortis interdum, turpis nibh consequat nisi, ut finibus nisl eros in nisi. Aenean convallis libero ut venenatis feugiat. Morbi in mi semper, fringilla metus sed, tincidunt mauris. In sed leo sit amet mi vehicula sodales. Aliquam sodales erat nibh, vitae sollicitudin magna suscipit gravida. Phasellus non fermentum massa. Integer in auctor tortor. Integer ut dolor eu dui faucibus efficitur eu et sem. Praesent rhoncus finibus pharetra. ", self.textEdit)
        self.display_paragraph("Curabitur dictum, sapien ac sagittis suscipit, ex sem sagittis sapien, vitae eleifend arcu augue sed massa. Aliquam aliquam eleifend scelerisque. Maecenas nec congue eros, a condimentum lorem. Maecenas dapibus vel nulla eu hendrerit. Quisque condimentum massa risus, quis tempus ex fermentum ut. Aliquam vel finibus metus. Proin mollis, sem vel porta tincidunt, mi eros cursus libero, in iaculis metus urna quis mi. Cras eu rhoncus quam. In posuere tempus molestie. Nunc pretium nibh lorem, a blandit massa vehicula vitae. Ut at ligula id odio tristique condimentum. Sed non neque consectetur, fringilla elit non, ultrices elit. Vestibulum nisi neque, suscipit sed sollicitudin sed, dictum in lacus. ", self.textEdit)
        self.display_paragraph("Donec elementum lorem in porta finibus. Sed lacinia egestas nisl ac vehicula. Phasellus consequat sapien tempor risus rhoncus blandit. In hac habitasse platea dictumst. Nulla facilisi. Nunc vitae nisi eget ipsum aliquam vulputate. Sed ut lorem sed felis fermentum finibus in a lectus. ", self.textEdit)
        self.display_paragraph("Proin pretium, magna sed feugiat pulvinar, nisi urna lacinia ligula, ut lacinia nisi lacus vitae libero. Etiam efficitur aliquam nisi, at rutrum massa blandit id. Pellentesque a magna auctor, sagittis tortor eget, suscipit lacus. Duis ac viverra justo. Interdum et malesuada fames ac ante ipsum primis in faucibus. Quisque vestibulum, nibh et porttitor imperdiet, nulla sapien mollis mauris, et ultricies dolor nibh id sem. Vivamus eleifend elit eget neque facilisis, vitae ullamcorper mi fringilla. Integer porttitor tortor nec magna pulvinar sodales. Phasellus lacus nibh, scelerisque iaculis mauris non, interdum finibus sem. Duis maximus risus nisl, eu semper massa egestas ac. Cras suscipit purus sit amet sodales auctor. ", self.textEdit)
        self.display_image("gui_utils/images/presentation_graph.png", self.textEdit)

        # Display image
        self.display_image("gui_utils/images/DIGI-Native-content-image.jpg", self.textEdit)

        # Tab 2
        self.tab2 = QWidget()
        self.notebook.addTab(self.tab2, "Analiza video")
        layout.addWidget(self.notebook)

        self.loadingLabel = QLabel("SE ÎNCARCĂ...", self)
        self.loadingLabel.setFont(QFont('Arial', 48))  
        self.loadingLabel.setAlignment(Qt.AlignCenter)
        self.loadingLabel.setFixedSize(self.size())  
        opacity_effect = QGraphicsOpacityEffect(self.loadingLabel)
        opacity_effect.setOpacity(1)  # Semi-transparent
        self.loadingLabel.setGraphicsEffect(opacity_effect)
        self.loadingLabel.setStyleSheet("background-color: rgba(0, 0, 0, 100); color: white;")
        self.loadingLabel.hide()

        self.blurEffect = QGraphicsBlurEffect(self)
        self.blurEffect.setBlurRadius(0)
        self.setGraphicsEffect(self.blurEffect)

        # Create an animation for the blur effect
        self.blurAnimation = QPropertyAnimation(self.blurEffect, b"blurRadius")
        self.blurAnimation.setDuration(10)  # The animation lasts 500 ms
        self.blurAnimation.setStartValue(0)  # The initial blur radius is 0
        self.blurAnimation.setEndValue(10)  # The final blur radius is 10
        self.blurAnimation.setLoopCount(1)  # Run the animation once

        # Add text to tab 2
        self.textEdit2 = QTextEdit(self.tab2)
        self.textEdit2.setReadOnly(True)
        self.textEdit2.setFont(QFont('Arial', 14))
        self.tab2Layout = QVBoxLayout(self.tab2)
        self.tab2Layout.addWidget(self.textEdit2)   


        self.display_paragraph("In cadrul acestei pagini se va realiza analiza videoului dorit. In josul paginii exista doua input-uri, primul pentru fisierul de input, unde veti selecta din sistemul de fisiere al computerului dumneavoastra fisierul dorit si al doilea, pentru denumirea fisierului de output care va fi adnotat cu ajutorul aplicatiei. Fisierul va fi pus intr-un folder predefinit si anume Licenta.", self.textEdit2)
        self.display_paragraph("Aplicatia are nevoie sa stie cate sensuri de mers exista in aceasta intersectie si cate zone mari de interes exista (de exemplu intr-o intersectie de tip 4-cornered exista 8 sensuri de mers si 4 zone mari de interes). De exemplu in poza de mai jos se poate observa cum sunt 4 zone de interes (numerele verzi) si 8 sensuri de mers (sagetile trasate cu rosu)", self.textEdit2)
        # Insert image
        self.display_image("gui_utils/images/Zone-Explanation.png", self.textEdit2)
        self.display_paragraph("Pentru a incepe analiza, apasati pe butonul 'Porneste analiza video'.", self.textEdit2)

        self.lineLayout = QHBoxLayout()

        # Add label to the layout
        self.outputFileLabel = QLineEdit(self.tab2)
        self.outputFileLabel.setText('Calea fisierului de input >')
        self.outputFileLabel.setReadOnly(True)
        # self.outputFileLabel.setFixedWidth(100)
        # Make with of the label as wide as the text
        self.outputFileLabel.setFixedWidth(self.outputFileLabel.fontMetrics().boundingRect(self.outputFileLabel.text()).width() + 10)
        self.outputFileLabel.setAlignment(Qt.AlignLeft)
        self.lineLayout.addWidget(self.outputFileLabel)

        # Create a QLineEdit
        self.outputFileLineEdit = QLineEdit(self.tab2)
        self.outputFileLineEdit.setPlaceholderText('Calea fisierului de input')
        self.lineLayout.addWidget(self.outputFileLineEdit)

        # Create a QToolButton
        self.inputFileButton = QToolButton(self.tab2)
        self.inputFileButton.setObjectName('inputFileButton')
        self.inputFileButton.setText('Alege fisierul de input')
        self.lineLayout.addWidget(self.inputFileButton)

        # Add the QHBoxLayout to the QVBoxLayout
        self.tab2Layout.addLayout(self.lineLayout)

        # Connect the button's clicked signal to the onInputFileButtonClicked slot
        self.inputFileButton.clicked.connect(self.onInputFileButtonClicked)

        self.lineLayout2 = QHBoxLayout()

        # Add a textbox to the tab 2 for introducing the output file name
        self.outputFileLabel2 = QLineEdit(self.tab2)
        self.outputFileLabel2.setText('Calea fisierului de output >')
        self.outputFileLabel2.setReadOnly(True)
        self.outputFileLabel2.setFixedWidth(self.outputFileLabel.fontMetrics().boundingRect(self.outputFileLabel2.text()).width() + 10)
        self.outputFileLabel2.setAlignment(Qt.AlignLeft)
        self.lineLayout2.addWidget(self.outputFileLabel2)

        self.outputFileLineEdit2 = QLineEdit(self.tab2)
        self.outputFileLineEdit2.setPlaceholderText('Numele fisierului de output, DOAR NUMELE, FARA EXTENSIE')
        self.lineLayout2.addWidget(self.outputFileLineEdit2)
        self.tab2Layout.addLayout(self.lineLayout2)
        self.outputFileLineEdit2.textChanged.connect(self.captureOutputFilename)

        self.lineLayoutZones = QHBoxLayout()
        # Add a textbox to the tab 2 for introducing the number of zones
        self.outputFileLabel3 = QLineEdit(self.tab2)
        self.outputFileLabel3.setText('Numarul de zone >')
        self.outputFileLabel3.setReadOnly(True)
        self.outputFileLabel3.setFixedWidth(self.outputFileLabel.fontMetrics().boundingRect(self.outputFileLabel3.text()).width() + 10)
        self.outputFileLabel3.setAlignment(Qt.AlignLeft)
        self.lineLayoutZones.addWidget(self.outputFileLabel3)

        self.outputFileLineEdit3 = QLineEdit(self.tab2)
        self.outputFileLineEdit3.setPlaceholderText('Numarul de zone (DOAR INTEGERS)')
        self.lineLayoutZones.addWidget(self.outputFileLineEdit3)
        self.outputFileLineEdit3.editingFinished.connect(lambda: self.captureNumberZones(self.outputFileLineEdit3.text()))

        self.tab2Layout.addLayout(self.lineLayoutZones)



        # Add a button to start the video analysis
        self.startButton = QToolButton(self.tab2)
        self.startButton.setObjectName('startButton')
        self.startButton.setText('Porneste analiza video')
        self.tab2Layout.addWidget(self.startButton)

        # When the button is clicked, the setup method is called
        self.startButton.clicked.connect(self.run_script)

        # Make a 3rd tab for the results
        self.tab3 = QWidget()
        self.notebook.addTab(self.tab3, "Rezultate")
        self.tab3Layout = QVBoxLayout(self.tab3)
        self.textEdit3 = QTextEdit(self.tab3)
        self.textEdit3.setReadOnly(True)
        self.textEdit3.setFont(QFont('Arial', 14))
        self.tab3Layout.addWidget(self.textEdit3)




    def display_paragraph(self, text, textEdit):
        cursor = QTextCursor(textEdit.document())
        cursor.movePosition(QTextCursor.End)
        blockFormat = QTextBlockFormat()
        blockFormat.setBottomMargin(10)
        blockFormat.setLineHeight(3.3, 4)
        blockFormat.setLeftMargin(5)
        cursor.insertBlock(blockFormat)
        cursor.insertText(text)

    def display_image(self, image_path, textEdit):
        image = QImage(image_path)
        # Resize the image
        cursor = QTextCursor(textEdit.document())
        cursor.movePosition(QTextCursor.End)

        # Create a block format for the image and set its alignment to center
        blockFormat = QTextBlockFormat()
        blockFormat.setAlignment(Qt.AlignCenter)  # Center the block
        blockFormat.setBottomMargin(10)
        cursor.insertBlock(blockFormat)

        # Now insert the image
        imageFormat = QTextImageFormat()
        imageFormat.setName(image_path)
        imageFormat.setWidth(700)
        cursor.insertImage(imageFormat)

        # If you want to add more text after the image, ensure to reset block formatting
        blockFormat = QTextBlockFormat()  # Reset to default formatting
        cursor.insertBlock(blockFormat)

    def onInputFileButtonClicked(self):
        # Get input file path
        filename, _ = QFileDialog.getOpenFileName(self, 'Open File', 'c:\\', 'Video Files (*.mp4 *.avi)')
        self.outputFileLineEdit.setText(filename)
        self.filename = filename

    def captureOutputFilename(self, text):
        # This method is called whenever the text in the QLineEdit changes
        self.output = text

    def captureNumberZones(self, text):
        # This method is called whenever the text in the QLineEdit changes

        try:
            self.zones = int(text)
        except ValueError:
            print("Please introduce an integer value")
            self.outputFileLineEdit3.clear()
            QMessageBox.warning(self, 'Eroare', 'Va rugam sa introduceti un numar intreg')


    def run_script(self):
        # Check if the input and output file paths are set
        if self.filename is None or self.output is None or self.zones is None:
            print("Va rugam sa completati toate campurile")
            QMessageBox.warning(self, 'EROARE', 'Va rugam sa completati toate campurile')
            return
        else:
            reply = QMessageBox.question(self, 'Message', 'Sunteti sigur ca datele introduse sunt corecte? Numel fisierului de input: ' + self.filename + ' Numel fisierului de output: ' + self.output + ' Numarul de zone: ' + str(self.zones), QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.Yes:
                QMessageBox.warning(self, 'ASTEPTATI', 'Va trebui sa alegeti zonele de interes din cadrul intersectiei. Apasati pe butonul "OK" pentru a continua')

                # Start the run in a separate thread
                self.loadingLabel.show()
                self.blurAnimation.start()

                self.runThread = QThread()
                self.runWorker = Run(self.filename, self.output, self.zones)
                self.runWorker.moveToThread(self.runThread)
                self.runThread.started.connect(self.runWorker.run)
                self.runWorker.finished.connect(self.on_run_finished)
                self.runWorker.displayWarning.connect(self.showWarning)
                self.runThread.start()

                # Cleanup
                self.runThread.finished.connect(self.runThread.quit)
                self.runThread.finished.connect(self.runThread.deleteLater)
                self.runWorker.finished.connect(self.runWorker.deleteLater)
            else:
                return
    @Slot()
    def on_run_finished(self):
        print("Wrapping up")
        # Stop the blur animation
        self.blurAnimation.stop()
        self.blurEffect.setBlurRadius(0)
        self.loadingLabel.hide()
        QMessageBox.information(self, 'Succes', 'Analiza video a fost finalizata cu succes. Rezultatele se gasesc in folderul Licenta')

        # Clear the input fields
        self.outputFileLineEdit.clear()
        self.outputFileLineEdit2.clear()
        self.outputFileLineEdit3.clear()

        self.filename = None
        self.output = None
        self.zones = None

        # Show the results
        self.display_paragraph("Rezultatele analizei video vor fi afisate aici", self.textEdit3)
        
        # Show the images in the tab 3
        self.display_paragraph("Cele mai normale traiectorii", self.textEdit3)
        if os.path.exists("gui_utils/results/average_trajectories.png"):
            self.display_image("gui_utils/results/average_trajectories.png", self.textEdit3)

        self.display_paragraph("Cele mai normale traiectorii pe un cadru din video", self.textEdit3)
        if os.path.exists("gui_utils/results/average_trajectories_on_frame.png"):
            self.display_image("gui_utils/results/average_trajectories_on_frame.png", self.textEdit3)

        self.display_paragraph("Primele 10 puncte ale traiectoriilor grupate", self.textEdit3)
        if os.path.exists("gui_utils/results/first_points.png"):
            self.display_image("gui_utils/results/first_points.png", self.textEdit3)

        self.display_paragraph("Ultimele 10 puncte ale traiectoriilor grupate", self.textEdit3)
        if os.path.exists("gui_utils/results/last_points.png"):
            self.display_image("gui_utils/results/last_points.png", self.textEdit3)

        self.display_paragraph("Primele 10 puncte din fiecare traiectorie grupate", self.textEdit3)
        if os.path.exists("gui_utils/results/first_trajectories_clustered.png"):
            self.display_image("gui_utils/results/first_trajectories_clustered.png", self.textEdit3)

        self.display_paragraph("Traiectoriile grupate per sensul de mers", self.textEdit3)
        # For every image in clusters folder, display it
        for image in os.listdir("gui_utils/results/clusters"):
            self.display_image(f"gui_utils/results/clusters/{image}", self.textEdit3)

        self.display_paragraph("Cele mai putin normale traiectorii per sensul de mers", self.textEdit3)
        # For every image in least_similar folder, display it
        for image in os.listdir("gui_utils/results/least_similar"):
            self.display_image(f"gui_utils/results/least_similar/{image}", self.textEdit3)

        self.display_paragraph("Traiectoriile medii per sensul de mers", self.textEdit3)
        if os.path.exists("gui_utils/results/average_trajectories.png"):
            self.display_image("gui_utils/results/average_trajectories.png", self.textEdit3)

        self.display_paragraph("Număru de vehicule ce trec dintr-o zonă în alta", self.textEdit3)
        if os.path.exists("gui_utils/results/vehicle_counts.png"):
            self.display_image("gui_utils/results/vehicle_counts.png", self.textEdit3)

        self.display_paragraph("Numărul de pietoni ce traversează strada", self.textEdit3)
        if os.path.exists("gui_utils/results/pedestrian_counts.png"):
            self.display_image("gui_utils/results/pedestrian_counts.png", self.textEdit3)

        # Reset the tab
        self.notebook.setCurrentIndex(2)
    
    @Slot()
    def showWarning(self):
        pass



app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()
