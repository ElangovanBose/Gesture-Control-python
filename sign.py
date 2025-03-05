import cv2
import mediapipe as mp
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import math
import time



detector = HandDetector(detectionCon=0.8,maxHands=1)

cap = cv2.VideoCapture(0)

#webcam size
wcam,hcam = 500,400

cap.set(3,wcam)
cap.set(4,hcam)

offset = 20
imgsize = 300

folder = "Data/C"
count = 0


while True:
    success, img = cap.read()

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]

        x,y,w,h = hand['bbox']

        imgwhite = np.ones((imgsize,imgsize,3),np.uint8)*255

        imgcrop = img[y-offset:y+h+offset,x-offset:x+w+offset]
        imgcropshape = imgcrop.shape

        

        aspectrat = h/w

        if aspectrat > 1:
            k = imgsize/h
            widthcal = math.ceil(k*w)
            imgresize = cv2.resize(imgcrop,(widthcal,imgsize))

            imgresizeshape = imgresize.shape

            widthgap = math.ceil((imgsize-widthcal)/2)
            imgwhite[:,widthgap:widthcal+widthgap] = imgresize

        else:
            k = imgsize/w
            heightcal = math.ceil(k*h)
            imgresize = cv2.resize(imgcrop,(imgsize,heightcal))

            imgresizeshape = imgresize.shape

            heightgap = math.ceil((imgsize-heightcal)/2)
            imgwhite[heightgap:heightcal+heightgap, :] = imgresize
            

        cv2.imshow('Input', imgcrop)
        cv2.imshow('white',imgwhite)

        

        
        

        
       

  
    cv2.imshow('first',img)

    c = cv2.waitKey(1)
    if c == 27:
        break
    if c == ord('s'):
        count +=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgwhite)
        print(count)

cap.release()
cv2.destroyAllWindows()
