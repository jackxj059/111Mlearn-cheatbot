import cv2
from cv2 import FILLED
import mediapipe as mp
import numpy as np
import mss
import keyboard
import time
import win32api
import win32con
import random
from pynput.mouse import Listener
mpPose = mp.solutions.pose
poses = mpPose.Pose(static_image_mode=True, model_complexity=0, enable_segmentation=True, smooth_landmarks=False,
                    min_detection_confidence=0.4)
output_model_path = "./model_1.xml"
mpDraw = mp.solutions.drawing_utils
poseLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=10)
poseconStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=10)
sct = mss.mss()
click = False
centerL = 0
centerT = 0
model = cv2.ml.SVM_load(output_model_path)
def cutObject(img):
    global  centerL, centerT
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3,3), np.uint8)
    th = cv2.dilate(th, kernel, iterations = 1)    
    contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours  )> 0:
        x,y,w,h = cv2.boundingRect(contours[-1])
    else:
        x,y,w,h = [0,0,0,0]
    return x,y,w,h

def predict(img):
    global model

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow("res",img)
    cv2.waitKey()
    chans = cv2.split(img)
    hists =  np.std([cv2.calcHist([chans[0]], [0], None, [255], [0, 250]),
                     cv2.calcHist([chans[1]], [0], None, [255], [0, 250]),
                     cv2.calcHist([chans[2]], [0], None, [255], [0, 250])],axis = 0).ravel().tolist()    
    x = np.array([hists], dtype=np.float32)
    Y = int(model.predict(x)[1][0][0])
    return Y

def getFrame():
    global monitor
    image = np.array(sct.grab(monitor))  # 截圖
    return image

def moveMouse(landmark):
    for i, lm in enumerate(landmark.landmark):
        if i == 11 and lm:
            leftlm = lm.x
        if i == 12 and lm:
            xPos = (lm.x + leftlm) * 0.5 * imgWide
            yPos = lm.y * imgHeigh
            xPos = int(xPos)
            yPos = int(yPos)
            # image = cv2.circle(image, (xPos, yPos), 4, (255, 0, 0), FILLED)
            x = int(int(((lm.x + leftlm) * 0.5 - 0.5) * imgWide * movespeed)/1)
            y = int(int((lm.y - 0.5) * imgHeigh * movespeed)/1)
            if (mouseleft < 0):
                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x, y, 0, 0)  
            break

def init(x, y, button, is_press):
    if is_press:
        global click, centerL, centerT
        centerL = x
        centerT = y
        click = True

def getPose(image):
    results = poses.process(image)
    if results.pose_landmarks:
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        bg = np.zeros_like(image)
        bg[bg == 0] = 255
        bg2 = bg.copy()
        mpDraw.draw_landmarks(bg2, results.pose_landmarks, mpPose.POSE_CONNECTIONS, poseLmsStyle, poseconStyle)
        image2 = np.where(condition, image, bg)
        x,y,w,h = cutObject(bg2)
        return [image2[y:y+h,x:x+w], results.pose_landmarks ]
    else :
        return None

if __name__ == '__main__':  
    framec=0
    listener = Listener(on_click=init)
    time.sleep(4)
    print("detection start")
    listener.start()
    while True:
        if click:
            listener.stop()
            print("successfully")
            break


    width = 350
    height = 400
    left = centerL - (width / 2)
    top = centerT - (height / 2)
    left = int(left)
    top = int(top)
    monitor = {"top": top, "left": left, "width": width, "height": height}  # 截圖方框位置
    istart = True
    movespeed = 0.5
    while True:
        mouseleft = win32api.GetKeyState(0x02)
        image = getFrame()  
        imgHeigh = image.shape[0]
        imgWide = image.shape[1]
        if True:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            res = getPose(image)
            if not res is None:
                img      = res[0]
                landmark = res[1]
                moveMouse(landmark)

    if False:
        while True:
            print("123")
            frame  = getFrame()
            cv2.imshow("getFrame",frame)
            key = cv2.waitKey()
            if key == ord('q'):
                break