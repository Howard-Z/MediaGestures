import mediapipe as mp
import cv2
import math
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from parse_hand_landmarker import parser
from get_root_dir import *

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = get_root_dir() + '/hand_landmarker.task'


def process_video(path):
    cap = cv2.VideoCapture(path)
    # Create a hand landmarker instance with the video mode:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        num_hands=2,
        running_mode=VisionRunningMode.VIDEO)
    with HandLandmarker.create_from_options(options) as landmarker:
        # The landmarker is initialized. Use it here.

        # Use OpenCV’s VideoCapture to load the input video.

        # Load the frame rate of the video using OpenCV’s CV_CAP_PROP_FPS
        # You’ll need it to calculate the timestamp for each frame.

        # Loop through each frame in the video using VideoCapture#read()
        fps = cap.get(cv2.CAP_PROP_FPS)
        output = np.empty(shape=(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 2, 63))
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            hand_landmarker_result = landmarker.detect_for_video(mp_image, math.floor((1000 / fps) * count))
            left, right = parser(handLandmarker=hand_landmarker_result)
            # output[count].append(left)
            # np.append(output[count][0], left)
            output[count][0] = left
            output[count][1] = right
            # np.append(output[count][1], right)
            # output[count].append(right)
            # print(len(output_left) == len(output_right))
            count += 1

        # Convert the frame received from OpenCV to a MediaPipe’s Image object.
    cap.release()
    return np.array(output)
    # cv2.destroyAllWindows()
