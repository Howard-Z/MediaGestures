import mediapipe as mp
import cv2
import math
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import sys
import os
base_path = os.curdir
sys.path.insert(0, base_path)
from utils.parse_hand_landmarker import parser

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = '/Users/howardzhu/Documents/git_repos/MediaGestures/hand_landmarker.task'


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image


def process_video(path):
    
    cap = cv2.VideoCapture(path)
    # Create a hand landmarker instance with the video mode:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        num_hands = 2,
        running_mode=VisionRunningMode.VIDEO)
    with HandLandmarker.create_from_options(options) as landmarker:
        # The landmarker is initialized. Use it here.
    
        # Use OpenCV’s VideoCapture to load the input video.

        # Load the frame rate of the video using OpenCV’s CV_CAP_PROP_FPS
        # You’ll need it to calculate the timestamp for each frame.

        # Loop through each frame in the video using VideoCapture#read()
        fps = cap.get(cv2.CAP_PROP_FPS)
        output = np.empty(shape = (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 2, 63))
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            hand_landmarker_result = landmarker.detect_for_video(mp_image, math.floor((1000/fps) * count))
            left, right = parser(handLandmarker=hand_landmarker_result)
            #output[count].append(left)
            # np.append(output[count][0], left)
            output[count][0] = left
            output[count][1] = right
            # np.append(output[count][1], right)
            #output[count].append(right)
            #print(len(output_left) == len(output_right))
            count += 1

        # Convert the frame received from OpenCV to a MediaPipe’s Image object.
    cap.release()
    return np.array(output)
    #cv2.destroyAllWindows()