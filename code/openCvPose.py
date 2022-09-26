import cv2
import mediapipe as mp
import numpy as np
import imutils
from matplotlib import pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

leftShoulder = []
rightShoulder = []

rightFoot = []
leftFoot = []

rightAnkle = []
leftAnkle = []

rightKnee = []
leftKnee =[]

nose = []

cap = cv2.VideoCapture('DakdakFalse.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
calc_timestamps = [0.0]

plt.ion()
figure, axis = plt.subplots(3,2,figsize=(8, 8))

result = True

bally = 0
ballx = 0

#cap = cv2.VideoCapture(0)

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            
            frame = imutils.resize(frame, width=350)
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
            
            #Detect the ball by the color blue

            img_HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

            BLUE_MIN = np.array([100,88,0],np.uint8)
            BLUE_MAX = np.array([141,255,255],np.uint8)
            
            frame_threshed = cv2.inRange(img_HSV, BLUE_MIN, BLUE_MAX)
            
            # Find contours
            cnts = cv2.findContours(frame_threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]  # Use index [-2] to be compatible to OpenCV 3 and 4

            # Draw rectangle around the ball
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                if (w*h) < 10000 and (w*h) > 1000:
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), thickness = 2)
                    bally  = y + h
                    ballx = x

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                #print(landmarks)
                timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
                calc_timestamps.append(calc_timestamps[-1] + 1000/fps)

                leftShoulder.append([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y])
                rightShoulder.append([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])

                rightFoot.append([landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y])
                leftFoot.append([landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y])

                rightAnkle.append([landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y])
                leftAnkle.append([landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y])

                rightKnee.append([landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y])
                leftKnee.append([landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y])

                nose.append([landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y])

                # get the size of the frame
                h, w, c = image.shape

                # Get the location of the knee
                kneex = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x) * w
                kneey = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y) * h

                # Check if the ball in the correct position comparing to the knee
                if(result and bally > kneey and kneex - 70 > ballx ) :
                    result = False
                
                X = calc_timestamps[1:]
                Y1 = [x[0] for x in rightShoulder]

                Y3 = [x[0] for x in leftShoulder]

                Y5 = [x[0] for x in rightAnkle]

                Y7 = [x[0] for x in leftAnkle]

                Y9 = [x[0] for x in rightKnee]

                Y11 = [x[0] for x in leftKnee]
                

                axis[0, 0].plot(X, Y1)
                axis[0, 0].set_title("right Shoulder X")

                axis[0, 1].plot(X, Y3)
                axis[0, 1].set_title("left Shoulder X")

                axis[1, 0].plot(X, Y5)
                axis[1, 0].set_title("rightAnkle X")

                axis[1, 1].plot(X, Y7)
                axis[1, 1].set_title("left Ankle X")

                axis[2, 0].plot(X, Y9)
                axis[2, 0].set_title("right Knee X")

                axis[2, 1].plot(X, Y11)
                axis[2, 1].set_title("left Knee X")


                plt.tight_layout()
                plt.draw()
                plt.pause(0.0001)

            except:
                pass
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    except:
            pass
    if(result) :
        print("The move is correct")
    else:
        print("The move is incorrect")


