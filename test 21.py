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
from time import sleep
from pynput.keyboard import Controller
from subprocess import call
from cvzone.ClassificationModule import Classifier
import math



def signlang():
        
        labels = ["A","B","C"]
        offset = 20
        imgsize = 300
        detector = HandDetector(detectionCon=0.8)
        classifier = Classifier("keras/keras_model.h5", "keras/labels.txt")

        cap1 = cv2.VideoCapture(0)
        while True:
            success, img1 = cap1.read()
            imgoutput = img1.copy()
            hands, img1 = detector.findHands(img1)

            if hands:
                hand = hands[0]

                x,y,w,h = hand['bbox']

                imgwhite = np.ones((imgsize,imgsize,3),np.uint8)*255

                imgcrop = img1[y-offset:y+h+offset,x-offset:x+w+offset]
                imgcropshape = imgcrop.shape

                

                aspectrat = h/w

                if aspectrat > 1:
                    k = imgsize/h
                    widthcal = math.ceil(k*w)
                    imgresize = cv2.resize(imgcrop,(widthcal,imgsize))

                    imgresizeshape = imgresize.shape

                    widthgap = math.ceil((imgsize-widthcal)/2)
                    imgwhite[:,widthgap:widthcal+widthgap] = imgresize

                    prediction, index = classifier.getPrediction(img1)
                    print(prediction,index)

                else:
                    k = imgsize/w
                    heightcal = math.ceil(k*h)
                    imgresize = cv2.resize(imgcrop,(imgsize,heightcal))

                    imgresizeshape = imgresize.shape

                    heightgap = math.ceil((imgsize-heightcal)/2)
                    imgwhite[heightgap:heightcal+heightgap, :] = imgresize

                    prediction, index = classifier.getPrediction(img1)
                    print(prediction,index)

                cv2.putText(imgoutput,labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
                    

                cv2.imshow('Input', imgcrop)
                cv2.imshow('white',imgwhite)

            cv2.imshow('img',imgoutput)

            esc = cv2.waitKey(1)
            if esc == ord('w'):
                break
        cap1.release()
        cv2.destroyAllWindows()

        return main()
        




def main():

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

            if cv2.waitKey(1) == ord('q'):
                break
        return

    class button():
        def __init__(self,pos,text,size=[85,85]):
            self.pos = pos
            self.size = size
            self.text = text

    

    def draws(img,buttonlist):

        for button in buttonlist:
            x, y = button.pos
            w, h = button.size                          #RGB colors
            cv2.rectangle(img, button.pos, (x+w,y+h), (255,0,255), cv2.FILLED)

                                #text size     #font       #font size  #font color
            cv2.putText(img,button.text,(x+18,y+60), cv2.FONT_HERSHEY_PLAIN,4,(255,255,255),4)

        return img
    
    hand = mp.solutions.hands
    hands = hand.Hands(static_image_mode=False,model_complexity=1,min_detection_confidence=0.75,min_tracking_confidence=0.75,max_num_hands=2)
    draw = mp.solutions.drawing_utils

    fin_id = [4,8,12,16,20]
    fintext = ""

    keyboard = Controller()

    #access speaker using pycaw
    device = AudioUtilities.GetSpeakers()
    interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))

    classifier = Classifier("keras/keras_model.h5", "keras/labels.txt")

    #volume range
    volmin, volmax = volume.GetVolumeRange()[:2]

    cap = cv2.VideoCapture(0)

    #webcam size
    wcam,hcam = 1300,800

    #screen width,screen height
    swidth,sheight = pyautogui.size()

    index_y = 0

    cap.set(3,wcam)
    cap.set(4,hcam)
    keys = [['Q','W','E','R','T','Y','U','I','O','P'],['A','S','D','F','G','H','J','K','L',';'],['Z','X','C','V','B','N','M',',','.','/']]        
    buttonlist = []
    for i in range(len(keys)):
        for j,key in enumerate(keys[i]):
            buttonlist.append(button([100*j+50,100*i+50],key))
    while True:
        
    
        success, img = cap.read()

        fheight,fwidth,_ = img.shape

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
                    #create a boundry box
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
                    
                    break
                
                    
                    

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


                            if (fingers[1] and fingers[2]) == 1 and (fingers[0] and fingers[3] and fingers[4]) == 0:
                                
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
                                    pyautogui.sleep(1)
                                    continue
                                    

                            if (fingers[0] and fingers[1]) == 0 and (fingers[2] and fingers[3] and fingers[4]) == 1 :
                                screen_record()


                        if label == 'Left':

                            if (fingers[1] and fingers[4]) == 1 and (fingers[0] and fingers[2] and fingers[3]) == 0 :
                                signlang()

                                            

                            if (fingers[1] and fingers[2]) == 1 and (fingers[0] and fingers[3] and fingers[4]) == 0 :
                                img = draws(img,buttonlist)

                                if lmList:
                                    for button in buttonlist:
                                        x, y = button.pos
                                        w, h = button.size

                                        if x < lmList[8][1] < x+w and y< lmList[8][2] < y+h:
                                        
                                        
                                            cv2.rectangle(img, button.pos, (x+w,y+h), (175,0,175), cv2.FILLED)

                                            cv2.putText(img,button.text,(x+18,y+60), cv2.FONT_HERSHEY_PLAIN,4,(255,255,255),4)
                                            x6, y6 = lmList[8][1],lmList[8][2]
                                            x7, y7 = lmList[12][1],lmList[12][2]

                                            cv2.circle(img, (x6,y6), 15, (255, 0, 0),cv2.FILLED)
                                            cv2.circle(img, (x7,y7), 15, (255, 0, 0),cv2.FILLED)

                                            #cv2.line(img, (x6,y6), (x7, y7), (255,0,0), 3)

                                            len7 = hypot(x7-x6,y7-y6)
                                            print(len7)

                                            #when click
                                            if len7<34:
                                                keyboard.press(button.text)
                                                cv2.rectangle(img, button.pos, (x+w,y+h), (0,255,0), cv2.FILLED)
                                                cv2.putText(img,button.text,(x+18,y+60), cv2.FONT_HERSHEY_PLAIN,4,(255,255,255),4)
                                                fintext += button.text
                                                sleep(0.15)

                                            
                                    cv2.rectangle(img, (50,350), (700,450), (175,0,175), cv2.FILLED)
                                    cv2.putText(img,fintext,(60,425), cv2.FONT_HERSHEY_PLAIN,4,(255,255,255),4)

               
      
        #show videos
        cv2.imshow('Input', img)

        c = cv2.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
        main()


