import cv2
import mediapipe as mp
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np
import screen_brightness_control as sbc

from cvzone.HandTrackingModule import HandDetector


hand = mp.solutions.hands
hands = hand.Hands()
draw = mp.solutions.drawing_utils

detector = HandDetector(detectionCon=0.8,maxHands=2)

#access speaker using pycaw
device = AudioUtilities.GetSpeakers()
interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

#volume range
volmin, volmax = volume.GetVolumeRange()[:2]

cap = cv2.VideoCapture(0)

#webcam size
wcam,hcam = 700,500

cap.set(3,wcam)
cap.set(4,hcam)


while True:
    success, img = cap.read()

    #convert bgr to rgb
    imgrgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result = hands.process(imgrgb)

    lmList = []
    #used for hand landmarks
    if result.multi_hand_landmarks:
        for handlandmark in result.multi_hand_landmarks:
            for id, lm in enumerate(handlandmark.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w),int(lm.y*h)
                lmList.append([id, cx, cy])
            #draw.draw_landmarks(img, handlandmark, hand.HAND_CONNECTIONS)

    if lmList != []:
        x1, y1 = lmList[4][1],lmList[4][2]
        x2, y2 = lmList[8][1],lmList[8][2]

        x3, y3 = lmList[4][1],lmList[4][2]
        x4, y4 = lmList[12][1],lmList[12][2]


        #for circle at top of finger
        cv2.circle(img, (x1,y1), 15, (255, 0, 0),cv2.FILLED)
        cv2.circle(img, (x2,y2), 15, (255, 0, 0),cv2.FILLED)

        cv2.circle(img, (x3,y3), 15, (255, 0, 0),cv2.FILLED)
        cv2.circle(img, (x4,y4), 15, (255, 0, 0),cv2.FILLED)


        #draw line between two fingers
        cv2.line(img, (x1,y1), (x2, y2), (255,0,0), 3)


        len1 = hypot(x2-x1,y2-y1)
        len2 = hypot(x4-x3,y4-y3)


        #convert hand range to volume range
        vol = np.interp(len1,[15, 220],[volmin, volmax])
        print(vol,len1)

        #for brightness control
        brightness = np.interp(len2,[15,220],[0,100])
        
        

        #set the master volume
        volume.SetMasterVolumeLevel(vol, None)

        #set brightness
        sbc.set_brightness(int(brightness))


    hands, img = detector.findHands(img)

    if hands:
        hand1 = hands[0]
        lmList1 = hand1['lmList']
        bbox1 = hand1['bbox']
        centerpoint1 = hand1['center']
        handtype1 = hand1['type']

        if len(hands) == 2:
            print('both')

        else:
            if handtype1 == 'Right':
                print('right')

            if handtype1 == 'Left':
                print('left')
    

    #show videos
    cv2.imshow('Input', img)

    c = cv2.waitKey(1)
    if c == 27:
        break


cap.release()
cv2.destroyAllWindows()
