import cv2
import mediapipe as mp
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np
import screen_brightness_control as sbc
import pyautogui
from datetime import datetime
import os
from google.protobuf.json_format import MessageToDict
from PIL import ImageGrab

from cvzone.HandTrackingModule import HandDetector

hand = mp.solutions.hands
hands = hand.Hands()
draw = mp.solutions.drawing_utils

detector = HandDetector(detectionCon=0.8,maxHands=2)

cap = cv2.VideoCapture(0)

#webcam size
wcam,hcam = 500,400

cap.set(3,wcam)
cap.set(4,hcam)


while True:
    success, img = cap.read()

    hands, img = detector.findHands(img)

    if hands:
        hand1 = hands[0]
        lmList1 = hand1['lmList']
        bbox1 = hand1['bbox']
        centerpoint1 = hand1['center']
        handtype1 = hand1['type']
        
        fin_count1 = detector.fingersUp(hand1)
        #l, info, img = detector.findDistance(lmList1[4], lmList1[8], img)
        print(lmList1[8])

        if len(hands) == 2:

            hand2 = hands[1]
            lmList2 = hand2['lmList']
            bbox2 = hand2['bbox']
            centerpoint2 = hand2['center']
            handtype2 = hand2['type']

            fin_count2 = detector.fingersUp(hand2)
            print(fin_count1,fin_count2)
            

        else:
            if handtype1 == 'Right':
                print(fin_count1)

            if handtype1 == 'Left':
                if fin_count1[0] == 1 and fin_count1[1] == 1:
                    #x3 = np.interp(lmList1[8],(0,1920),(0,1920))
                    #y3 = np.interp(lmList1[12],(0,1080),(0,1080))
                    #l, info, img = detector.findDistance(lmList1[4], lmList1[8], img)
                    #print(l)
                    continue
                print(fin_count1)
    

  
    cv2.imshow('Input', img)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
