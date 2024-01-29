def parser(handLandmarker):
    if handLandmarker.hand_landmarks == []:
        return None
    llandmarks = handLandmarker.hand_landmarks[0]
    llist = [coord for landmark in llandmarks for coord in [landmark.x, landmark.y, landmark.z]]
    if len(handLandmarker.hand_landmarks) == 2:
        rlandmarks = handLandmarker.hand_landmarks[1]
        rlist = [coord for landmark in rlandmarks for coord in [landmark.x, landmark.y, landmark.z]]
    else:
        rlist = None
    return llist,rlist