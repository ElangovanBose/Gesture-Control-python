import cv2
import mediapipe as mp
import HandTracking as ht

hand = mp.solutions.hands
hands = hand.Hands()
draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

#webcam size
wcam,hcam = 500,400

cap.set(3,wcam)
cap.set(4,hcam)

detector = ht.handDetector(maxHands = 2)
while True:
    success, img = cap.read()
   # imgrgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
   # result = hands.process(imgrgb)

    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    if len(lmList) != 0:
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]
        #print(x1,y1,x2,y2)

        fingers = detector.fingersUp()
        print(fingers)
        
  #  if result.multi_hand_landmarks:
 #       for handlandmark in result.multi_hand_landmarks:
 #           draw.draw_landmarks(img, handlandmark, hand.HAND_CONNECTIONS)
    cv2.imshow('Input', img)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
