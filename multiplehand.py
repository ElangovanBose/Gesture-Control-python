import cv2
import mediapipe as mp

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

        if len(hands) == 2:
            print('both')

        else:
            if handtype1 == 'Right':
                print('right')

            if handtype1 == 'Left':
                print('left')
    

  
    cv2.imshow('Input', img)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
