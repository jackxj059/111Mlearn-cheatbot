import cv2
import mediapipe as mp
from PIL import ImageGrab
import numpy as np
import pyautogui
# conn = mp.solutions.pose.POSE_CONNECTIONS
pose = mp.solutions.pose.Pose(min_detection_confidence=0.5,enable_segmentation=True, min_tracking_confidence=0.5)
# mp_drawing = mp.solutions.drawing_utils
# spec = mp.solutions.drawing_styles.get_default_pose_landmarks_style()
output_model_path = "./model_1.xml"
model = cv2.ml.SVM_load(output_model_path)
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
    frame = np.array(ImageGrab.grab())
    frame= frame[100:500,:]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def getPose(image):
    global pose
    results = pose.process(frame)
    bg = np.zeros_like(frame)
    bg[bg==0] = 255
    
     
    if results.pose_landmarks:
        # mp_drawing.draw_landmarks(image, results.pose_landmarks, conn, spec)
        try:
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            image = np.where(condition, image, bg)
            gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, th1 = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
            th1 = cv2.GaussianBlur(th1, (5, 5), 0)
            kernel = np.ones((11,11), np.uint8)
            th1 = cv2.dilate(th1, kernel, iterations = 1)
            # cv2.imshow('th2', th1)

            contours, hierarchy = cv2.findContours(th1, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            center=[]
            lastX=0
            lastY=0
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if not lastX == 0:
                    cv2.line(th1, (x, y), (lastX, lastY), 255, 5)
                lastY = y
                lastX = x
            contours, hierarchy = cv2.findContours(th1, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            x, y, w, h = cv2.boundingRect(contours[0])
            img = image[y:y+h,x:x+w]
            return  x, y, w, h, img
        except Exception as e:
            print("except")
        return ""
    else:
        return ""
if __name__ == '__main__':  
    framec=0

    if True:
        while True:
            frame = getFrame()
            location = getPose(frame)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            if len(location) > 1: 
                x, y, w, h, img = location

                res = predict(img)
            else:
                continue
            cv2.imwrite('./outside/' + str(int(res)) + "/" +str(framec) + ".jpg", img)
            framec += 1
            print(str(res),"    :" ,x, y, w, h   )    
            # cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
    if False:
        while True:
            print("123")
            frame  = getFrame()
            cv2.imshow("getFrame",frame)
            key = cv2.waitKey()
            if key == ord('q'):
                break