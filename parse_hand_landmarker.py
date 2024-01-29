def parser(handLandmarker):
    if handLandmarker.hand_landmarks == []: #makes sure it's not empty
        return None
    
    llandmarks = handLandmarker.hand_landmarks[0]
    llist = [coord for landmark in llandmarks for coord in [landmark.x, landmark.y, landmark.z]] #left hand list
    
    if len(handLandmarker.hand_landmarks) == 2: #checks right hand list exists
        rlandmarks = handLandmarker.hand_landmarks[1]   #same process
        rlist = [coord for landmark in rlandmarks for coord in [landmark.x, landmark.y, landmark.z]]
    else:
        rlist = None #else make sure it's none
    return llist, rlist #return both as tuple