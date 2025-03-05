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
import HandTrackingModule as htm

def count(arr):
    s = ""
    for i in arr:
        s+= str(arr[i]);

    if (s == "00000"):
        return 0
    elif (s == "01000"):
        return 1
    elif (s == "01100"):
        return 2
    elif (s == "01110"):
        return 3
    elif (s == "01111"):
        return 4
    elif (s=="11000"):
        print(5)

hand = mp.solutions.hands
hands = hand.Hands()
draw = mp.solutions.drawing_utils

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
            draw.draw_landmarks(img, handlandmark, hand.HAND_CONNECTIONS)

    tipid = [4,8,12,16,20]

    if lmList != []:

        fingers = []

        #thumb
        if(lmList[tipid[0]][1] > lmList[tipid[0]-1][1]):
            fingers.append(1)
        else:
            fingers.append(0)


        #4 fingers
        for id in range(1,len(tipid)):
            if(lmList[tipid[id]][2]<lmList[tipid[id]-2][2]):
                #print(lmList[tipid[id]][2],lmList[tipid[id]-2][2])
                fingers.append(1)

            else:
                fingers.append(0)

        cfingers = count(fingers)

        if cfingers == 1:

            #take screenshot
            screenshot = pyautogui.screenshot()

            #get a system time
            now = datetime.now()
            curr_time = now.strftime("%H%M%S")

            #convert current time to string
            str_curr_time = str(curr_time)
            fname = str_curr_time+".png"

            #save a screenshot
            screenshot.save(fname)

            #check the file that already exists
            check = os.path.isfile(fname)
            if check == True:
                continue

        if cfingers == 2:

            resolution = pyautogui.size()

            codec = cv2.VideoWriter_fourcc(*"MJPG")

            #get a system time
            now = datetime.now()
            curr_time = now.strftime("%H%M%S")

            #convert current time to string
            str_curr_time = str(curr_time)
            fname = str_curr_time+".avi"

            fps = 20.0

            out = cv2.VideoWriter(fname,codec,fps,resolution)

            while cfingers != 3:
                ss = pyautogui.screenshot()
                frame = np.array(ss)

                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

                out.write(frame)

            out.release()
            cv2.destroyAllWindows()
        
        x1, y1 = lmList[4][1],lmList[4][2]
        x2, y2 = lmList[8][1],lmList[8][2]

        x3, y3 = lmList[4][1],lmList[4][2]
        x4, y4 = lmList[12][1],lmList[12][2]

        if lmList[8][2] < lmList[6][2]:

            #for circle at top of finger
            cv2.circle(img, (x1,y1), 15, (255, 0, 0),cv2.FILLED)
            cv2.circle(img, (x2,y2), 15, (255, 0, 0),cv2.FILLED)

            #draw line between fingers
            cv2.line(img, (x1,y1), (x2, y2), (255,0,0), 3)

            #calculate length between fingers
            len1 = hypot(x2-x1,y2-y1)

            #change finger range to volume range
            vol = np.interp(len1,[15, 220],[volmin, volmax])
            #print(vol,len1)

            #set volume
            volume.SetMasterVolumeLevel(vol, None)


        if lmList[12][2] < lmList[10][2]:

            #for circle at top of fingers
            cv2.circle(img, (x3,y3), 15, (255, 0, 0),cv2.FILLED)
            cv2.circle(img, (x4,y4), 15, (255, 0, 0),cv2.FILLED)

            #draw line between fingers
            cv2.line(img, (x3,y3), (x4,y4), (255,0,0), 3)

            #calculate length between fingers
            len2 = hypot(x4-x3,y4-y3)

            brightness = np.interp(len2,[15,220],[0,100])

            #set brightness
            sbc.set_brightness(int(brightness))


        

    #show videos
    cv2.imshow('Input', img)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
