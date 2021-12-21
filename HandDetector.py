import cv2
import mediapipe as mp
import switch as switch


class HandDetactor:
    def __init__(self,mode=False,nbrHands=2,detactConf=0.5,trackConf=0.5):
        self.mode=mode
        self.nbrHands=nbrHands
        self.detactConf=detactConf
        self.trackConf=trackConf
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.nbrHands,1,self.detactConf,self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils
    def findHands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    def findPosHand(self,img,nbsHand=0,draw=True):
        lmList=[]
        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[nbsHand]
            for id,lm in enumerate(myHand.landmark):
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
        return lmList
    def findFinger(self,img,nbsFinger,nbsHand,draw=True):
        fingerLmList=[]
        if nbsFinger==0:
            id=[2, 3, 4]
        if nbsFinger==1:
            id=[6, 7, 8]
        if nbsFinger == 2:
            id=[10, 11, 12]
        if nbsFinger == 3:
            id = [14, 15, 16]
        if nbsFinger == 4:
            id = [18, 19, 20]
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[nbsHand]
            for i, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if i ==id[0] or i ==id[1] or i ==id[2] :
                    fingerLmList.append([i,cx,cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return fingerLmList





