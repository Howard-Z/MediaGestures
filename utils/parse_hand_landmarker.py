import numpy as np


def parser1(handLandmarker):
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

def parser(handLandmarker):
    if not handLandmarker.hand_world_landmarks:  # makes sure it's not empty
        return None, None
    
    left, right = None, None
    hands = handLandmarker.handedness
    if len(hands) == 1:
        #print("1 hand")
        hand = hands[0][0].category_name
        llandmarks = handLandmarker.hand_world_landmarks[0]
        llist = [coord for landmark in llandmarks for coord in [landmark.x, landmark.y, landmark.z]]
        if hand == "Left":
            left = np.array(llist)
            #print("set left", left)
        else:
            right = np.array(llist)
            #print("set right", right)
    elif len(hands) == 2:
        #print("2 hands")
        for category in hands:
            category = category[0]
            index = category.index
            hand = category.category_name
            llandmarks = handLandmarker.hand_world_landmarks[index]
            llist = [coord for landmark in llandmarks for coord in [landmark.x, landmark.y, landmark.z]]
            if hand == "Left":
                left = np.array(llist)
                #print("set left!", left)
            else:
                right = np.array(llist)
                #print("set right!", right)
    return left, right