import mediapipe as mp
import cv2
import math
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from utils.get_root_dir import get_root_dir
import numpy as np

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = get_root_dir() + '/hand_landmarker.task'

cap = cv2.VideoCapture('./raw_videos/pinch/trimmed_2024-01-30_02-13-07.mp4')

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


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
        x_coordinates, y_coordinates = [(landmark.x, landmark.y) for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image


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
    # frame_gap = (1000//fps)
    frame_gap = 1
    print(frame_gap)
    ts = 0
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        if cv2.waitKey(int(frame_gap)) == ord('q'):
            break
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        hand_landmarker_result = landmarker.detect_for_video(mp_image, math.floor((1000 / fps) * count))
        print(hand_landmarker_result)
        annotated_image = draw_landmarks_on_image(frame, hand_landmarker_result)
        cv2.imshow('frame', annotated_image)
        count += 1

    # Convert the frame received from OpenCV to a MediaPipe’s Image object.
cap.release()
cv2.destroyAllWindows()
