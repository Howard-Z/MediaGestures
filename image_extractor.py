import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from utils.parse_hand_landmarker import *

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = 'C:/Users/Howard/Documents/Git_Repos/MediaGestures/hand_landmarker.task'

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

# Create a hand landmarker instance with the image mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands = 2,
    running_mode=VisionRunningMode.IMAGE)
with HandLandmarker.create_from_options(options) as landmarker:
    img = cv2.imread("test.jpg")
    img = cv2.resize(img, (640, 480))
    #mp_image = mp.Image.create_from_file('/test.jpg')
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    hand_landmarker_result = landmarker.detect(mp_image)
    annotated_image = draw_landmarks_on_image(img, hand_landmarker_result)
    print(hand_landmarker_result)
    left, right = parser(handLandmarker= hand_landmarker_result)
    left = np.array(left)
    right = np.array(right)
    print("left")
    if left is None:
       print("Left is None")
    else:
       print(left)
       print("length = ", len(left))

    print("right")
    if right is None:
       print("Right is None")
    else:
       print(right)
       print("length = ", len(right))
    for i in range(0, len(left), 3):
       output = "("
       output += str(left[i])
       output += ", "
       output += str(left[i + 1])
       output += ", "
       output += str(left[i + 2])
       # output += "0.0"
       output += ")"
       print(output)
    
    print("right")
    for i in range(0, len(right), 3):
       output = "("
       output += str(right[i])
       output += ", "
       output += str(right[i + 1])
       output += ", "
       output += str(right[i + 2])
       # output += "0.0"
       output += ")"
       print(output)

    # cv2.imshow("AHHH", annotated_image)
    # k = cv2.waitKey(0) & 0xFF
    
    