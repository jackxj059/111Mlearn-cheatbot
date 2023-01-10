import cv2
import numpy as np
import os 
# 
# SAMPLE_PATH = "./result/good/"
SAMPLE_PATH = "./result1/"
output_model_path = "./model_2.xml"
upper  = np.array([105, 155,197])
lowwer = np.array([60, 115,157])
test = "./result/good/2/1414.jpg"
def _train():
    x=[]
    y=[]
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)  # Set SVM type
    svm.setKernel(cv2.ml.SVM_LINEAR)  # Set SVM Kernel
    svm.setC(0.1)  # Set parameter C
    svm.setGamma(1.0)  # Set parameter Gamma
    for category in os.listdir(SAMPLE_PATH):
        print(category)
        for image in os.listdir(SAMPLE_PATH+"/"+category):
            img = cv2.imread(SAMPLE_PATH +category+"/" + image)
            img - cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            chans = cv2.split(img)
            hists =  np.std([cv2.calcHist([chans[0]], [0], None, [255], [0, 250]),
                             cv2.calcHist([chans[1]], [0], None, [255], [0, 250]),
                             cv2.calcHist([chans[2]], [0], None, [255], [0, 250])],axis = 0).ravel().tolist()
            x += [hists]
            y += category
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print("x",x.shape)
    print("y",y.shape)
    svm.trainAuto(x, cv2.ml.ROW_SAMPLE, y)
    svm.save(output_model_path)
def train():
    x=[]
    y=[]
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)  # Set SVM type
    svm.setKernel(cv2.ml.SVM_LINEAR)  # Set SVM Kernel
    svm.setC(0.1)  # Set parameter C
    svm.setGamma(1.0)  # Set parameter Gamma
    for category in os.listdir("./result1"):
        if category == str(3):
            continue
        for image in os.listdir("./result1/"+category):
            img = cv2.imread("./result1/"+category+"/" + image)
            desc =  np.array([getMask_1(img),getMask_2(img)]).ravel().tolist()
            x += [desc]
            y += category
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print("x",x.shape)
    print("y",y.shape)
    svm.trainAuto(x, cv2.ml.ROW_SAMPLE, y)
    svm.save(output_model_path)
def getMask_1(img):
    upper  = np.array([105, 160,200])
    lowwer = np.array([60, 100,130]) 
    mask = cv2.inRange(img,lowwer, upper)
    num = np.count_nonzero(mask)
    all = mask.shape[0]*mask.shape[1]
    return num / all
    
def getMask_2(img):
    upper  = np.array([100, 60, 50])
    lowwer = np.array([50,  30, 10])
    mask = cv2.inRange(img,lowwer, upper)
    num = np.count_nonzero(mask)
    all = mask.shape[0]*mask.shape[1]   
    return num / all
def trainvalid():
    x=[]
    y=[]
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)     # Set SVM type
    svm.setKernel(cv2.ml.SVM_RBF)  # Set SVM Kernel
    svm.setC(0.1)                     # Set parameter C
    svm.setGamma(1.0)                 # Set parameter Gamma    
    for category in os.listdir("result1"):
        if int(category) == 3:
            results = '5'
            print("unvalid")
        else:
            results = '3'
            print("valid")

        for image in os.listdir("result1/"+category):
            img = cv2.imread("result1/" +category+"/" + image)
            ratio = img.shape[0]/img.shape[1]
            x.append(ratio)
            y += results
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print("x",x.shape)
    print("y",y.shape)
    svm.trainAuto(x, cv2.ml.ROW_SAMPLE, y)
    svm.save("valid.xml")
    

def test(path):
    def getColor(event, x, y, flags, para):
        if event == cv2.EVENT_LBUTTONDOWN:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            print("PIX :",x,y)
            print("BGR :",img[y,x])
            print("GRAY:",gray[y,x])
            print("HSV :",hsv[y,x])
    # upper  = np.array([105, 160,200])    
    # lowwer = np.array([60, 100,130])    

    #____________ skin _______________#
    upper  = np.array([100, 60, 50])
    lowwer = np.array([50,  30, 10]) 
    img = cv2.imread(path)
    cv2.namedWindow("img")
    cv2.setMouseCallback("img", getColor)
    # cv2.imshow('range', mask)
    # cv2.imshow('img', img)
    # cv2.waitKey() 
    if True:
        for category in os.listdir("./result1"):
            if category == str(3) :
                continue
            print("===========================",str(category),"===========================")
            for image in os.listdir( "./result1/"+category):

                img = cv2.imread( "./result1/" + category +"/" + image)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(img,lowwer, upper)
                num = np.count_nonzero(mask)
                all = mask.shape[0]*mask.shape[1]
                
                print(num / all)
 
def predict(img):
    model = cv2.ml.SVM_load(output_model_path)
    x = np.array([[getMask_1(img),getMask_2(img)] ], dtype=np.float32)   
    Y = model.predict(x)[1][0][0]
    return int(Y)
def isValid():

    model = cv2.ml.SVM_load("valid.xml")
    for category in os.listdir("result1"):
        print("_______________",str(category) ,"_________")
        for image in os.listdir("result1/"+category):
            img = cv2.imread("result1/" +category+"/" + image)
            ration = img.shape[0]/img.shape[1]
            ration= np.array([ration], dtype=np.float32)
            Y = model.predict(ration)[1][0][0]
            print(Y)
            cv2.imwrite("./outside/"+ str(int(Y)) +'/'+ image,img )
    return str(int(Y))   
     
if __name__ == "__main__":
    # test('result1/2/4.jpg')
    # train()
    # trainvalid()
    # isValid()
    # test('./result/good/1/16.jpg')
    # test('./result/good/1/16.jpg')
    # train()
    # for category in os.listdir(SAMPLE_PATH):
    #     if category == str(3):
    #         continue
    for image in os.listdir("./outside/3/"):
        img = cv2.imread("./outside/3/" + image)
        # if not int(category) == int(predict(img)):
        cv2.imwrite("./fix/"+str(predict(img))+"/"+ image, img)
        # print(category ,int(predict(img)))
            
        



