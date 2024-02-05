import numpy as np


def parser(handLandmarker):
    if not handLandmarker.hand_world_landmarks:  # makes sure it's not empty
        return None, None

    llandmarks = handLandmarker.hand_world_landmarks[0]
    llist = [coord for landmark in llandmarks for coord in [landmark.x, landmark.y, landmark.z]] # left hand list
    
    if len(handLandmarker.hand_world_landmarks) == 2: # checks right hand list exists
        rlandmarks = handLandmarker.hand_world_landmarks[1]   # same process
        rlist = [coord for landmark in rlandmarks for coord in [landmark.x, landmark.y, landmark.z]]
    else:
        rlist = None # else make sure it's none
    return np.array(llist), np.array(rlist) # return both as tuple