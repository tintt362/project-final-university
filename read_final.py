import os
import argparse
import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter
from scipy.spatial import distance as dist
from collections import OrderedDict
import threading
import multiprocessing
import time

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', default='custom_model_lite/custom_model_lite')
parser.add_argument('--graph', default='detect.tflite')
parser.add_argument('--labels', default='car_label_map.pbtxt')
parser.add_argument('--threshold', default=0.3, type=float)
parser.add_argument('--video', default='output1_video.mp4')
parser.add_argument('--edgetpu', action='store_true')
args = parser.parse_args()

# Paths
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, args.modeldir, args.graph)
PATH_TO_LABELS = os.path.join(CWD_PATH, args.modeldir, args.labels)
VIDEO_PATH = os.path.join(CWD_PATH, args.video)

# Load labels
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
if labels[0] == '???':
    del labels[0]

# Load TensorFlow Lite model
interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height, width = input_details[0]['shape'][1], input_details[0]['shape'][2]
floating_model = (input_details[0]['dtype'] == np.float32)

# Get model output index
outname = output_details[0]['name']
boxes_idx, classes_idx, scores_idx = (1, 3, 0) if 'StatefulPartitionedCall' in outname else (0, 1, 2)

# Draw lines and add text on the frame
def draw_lines_with_text(frame, points, color, text, text_position):
    for i in range(len(points) - 1):
        cv2.line(frame, points[i], points[i + 1], color, 2)
    cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# Check if a point is inside a polygon (region)
def point_in_polygon(point, polygon):
    point = (int(point[0]), int(point[1]))
    return cv2.pointPolygonTest(np.array(polygon, np.int32), point, False) >= 0

# Centroid Tracker class
class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (xmin, ymin, xmax, ymax)) in enumerate(rects):
            cX = int((xmin + xmax) / 2.0)
            cY = int((ymin + ymax) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            for row in set(range(0, D.shape[0])).difference(usedRows):
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            for col in set(range(0, D.shape[1])).difference(usedCols):
                self.register(inputCentroids[col])

        return self.objects

# Initialize the centroid tracker
ct = CentroidTracker(maxDisappeared=40)

# Define regions (polygons) for each "Vùng"
regions = {
    "Vung 1": [(1, 337), (117, 322), (132, 388), (3, 410)],
    "Vung 2": [(157, 264), (282, 249), (168, 139), (93, 143)],
    "Vung 3": [
    (447, 258), (496, 294), (575, 281), (639, 275),
    (639, 236), (595, 241), (537, 248), (452, 255)
],

    "Vung 4": [(535, 354), (373, 384), (453, 477), (635, 472), (638, 443)]
}

# Initialize total car counters for each region
total_counts = {region: 0 for region in regions}
counted_cars = {region: set() for region in regions}
current_cars_in_region = {region: 0 for region in regions}

# Tạo từ điển để theo dõi vùng đã đếm cho mỗi xe
object_to_region = {}

# Function to process the model in a separate thread
def process_model(interpreter, input_data, input_details, output_details, boxes_idx, classes_idx, scores_idx, args, imH, imW, results_queue):
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

    rects = []
    for i in range(len(scores)):
        if args.threshold < scores[i] <= 1.0:
            ymin, xmin, ymax, xmax = int(boxes[i][0] * imH), int(boxes[i][1] * imW), int(boxes[i][2] * imH), int(boxes[i][3] * imW)
            rects.append((xmin, ymin, xmax, ymax))

    results_queue.put(rects)

# Function to update centroid tracking in a separate thread
def track_objects(ct, rects, current_counts, regions, counted_cars, total_counts, frame, object_to_region):
    objects = ct.update(rects)

    # Reset the current count for all regions at the start of processing each frame
    for region_name in regions:
        current_counts[region_name] = 0

    # Loop through each detected object (car) and track it across the regions
    for (objectID, centroid) in objects.items():
        text = f"ID {objectID}"
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        for region_name, polygon in regions.items():
            # Kiểm tra nếu object đã được đếm trong vùng khác
            if objectID in object_to_region:
                # Nếu object đã đếm ở một vùng khác, bỏ qua các vùng khác
                if object_to_region[objectID] != region_name:
                    continue

            if point_in_polygon(centroid, polygon):  # If the object is in the region
                current_counts[region_name] += 1
                if objectID not in counted_cars[region_name]:
                    total_counts[region_name] += 1  # Increase the count if not already counted
                    counted_cars[region_name].add(objectID)
                    object_to_region[objectID] = region_name  # Ghi lại vùng mà object đã đếm
            elif objectID in counted_cars[region_name]:  # If the object has left the region
                counted_cars[region_name].remove(objectID)

    # Reset the total count to 0 when no cars are in the region
    for region_name in regions:
        if current_counts[region_name] == 0:
            total_counts[region_name] = 0  # Reset to 0 when no cars remain

# Function to handle video processing
def process_video(video_path, results_queue):
    video = cv2.VideoCapture(video_path)
    imW, imH = video.get(cv2.CAP_PROP_FRAME_WIDTH), video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    if imW == 0 or imH == 0:
        imW, imH = 640, 480

    return video, imW, imH

# Main function to process video frames
def main():
    results_queue = multiprocessing.Queue()

    video, imW, imH = process_video(VIDEO_PATH, results_queue)

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            print("Cannot read video")
            break

        frame_resized = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)
        if floating_model:
            input_data = (np.float32(input_data) - 127.5) / 127.5

        model_thread = threading.Thread(target=process_model, args=(interpreter, input_data, input_details, output_details, boxes_idx, classes_idx, scores_idx, args, imH, imW, results_queue))
        model_thread.start()

        rects = results_queue.get()

        tracker_thread = threading.Thread(target=track_objects, args=(ct, rects, current_cars_in_region, regions, counted_cars, total_counts, frame, object_to_region))
        tracker_thread.start()

        # Draw the regions
        draw_lines_with_text(frame, regions["Vung 1"], (0, 255, 0), "Vung 1*", (50, 300))
        draw_lines_with_text(frame, regions["Vung 2"], (255, 0, 255), "Vung 2*", (180, 260))
        draw_lines_with_text(frame, regions["Vung 3"], (0, 255, 255), "Vung 3*", (500, 250))
        draw_lines_with_text(frame, regions["Vung 4"], (255, 0, 0), "Vung 4*", (400, 400))

        # Display the car counts in each region
        for idx, (region_name, total) in enumerate(total_counts.items()):
            cv2.putText(frame, f"{region_name} Total: {total}", (10, 30 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow('Object detector and tracker', frame)
        if cv2.waitKey(1) == ord('q'):
            break

        model_thread.join()
        tracker_thread.join()

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
