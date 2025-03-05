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


def screen_record():
    image = ImageGrab.grab()
    imgarr = np.array(image)

    shape = imgarr.shape

    now = datetime.now()
    curr_time = now.strftime("%H%M%S")

    #convert current time to string
    str_curr_time = str(curr_time)
    fname = str_curr_time+".avi"

    video_writer = cv2.VideoWriter(fname,cv2.VideoWriter_fourcc('M','J','P','G'),50,(shape[1], shape[0]))

    while True:
        image = ImageGrab.grab()

        imgarr = np.array(image)

        final_img = cv2.cvtColor(imgarr,cv2.COLOR_RGB2BGR)

        video_writer.write(final_img)

        if cv2.waitKey(1) == ord('e'):
            continue
    video_writer.release()
    return

        
        
    
hand = mp.solutions.hands
hands = hand.Hands(static_image_mode=False,model_complexity=1,min_detection_confidence=0.75,min_tracking_confidence=0.75,max_num_hands=2)
draw = mp.solutions.drawing_utils

fin_id = [4,8,12,16,20]

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
    xlist = []
    ylist = []
    bbox = []
    #used for hand landmarks
    if result.multi_hand_landmarks:
        myhand = result.multi_hand_landmarks[0]
        for handlandmark in result.multi_hand_landmarks:
            if True:
                draw.draw_landmarks(img, handlandmark, hand.HAND_CONNECTIONS)
                
            for id, lm in enumerate(myhand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w),int(lm.y*h)
                lmList.append([id, cx, cy])
                xlist.append(cx)
                ylist.append(cy)

                if True:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xlist), max(xlist)
            ymin, ymax = min(ylist), max(ylist)
            bbox = xmin, ymin, xmax, ymax     

            if True:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

            

            fingers = []
            if lmList[fin_id[0]][1] > lmList[fin_id[0]-1][1]:
                fingers.append(1)

            else:
                fingers.append(0)

            for id in range(1,5):
                if lmList[fin_id[id]][2] < lmList[fin_id[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

                    

                
            print(fingers)
            if len(result.multi_handedness) == 2:
                print("both")

            else:
                 for i in result.multi_handedness:
                     
                     label = MessageToDict(i)['classification'][0]['label']
                     
                     if label == 'Right':

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


                     if label == 'Left':
                         continue
                            
                    
                



        if fingers == 1:

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

        if fingers == 1000:

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

            
            while True:
                ss = pyautogui.screenshot()
                frame = np.array(ss)

                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

                out.write(frame)
                if len(result.multi_handedness) == 2:
                    out.release()
                    cv2.destroyAllWindows()
        

    #show videos
    cv2.imshow('Input', img)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
