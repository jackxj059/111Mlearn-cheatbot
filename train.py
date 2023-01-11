import cv2
import numpy as np
import os 
# 
# SAMPLE_PATH = "./result/good/"
SAMPLE_PATH = "./trainSet/"
output_model_path = "./model_2.xml"
upper  = np.array([105, 155,197])
lowwer = np.array([60, 115,157])
test = "./result/good/2/1414.jpg"
def train():
    x=[]
    y=[]
    svm = cv2.ml.SVM_create()        #  ┐
    svm.setType(cv2.ml.SVM_C_SVC)    #    
    svm.setKernel(cv2.ml.SVM_INTER)  #  創建SVM模型，並設定參數  
    svm.setC(0.1)                    #    
    svm.setGamma(1.0)                #  ┘    

    for category in os.listdir(SAMPLE_PATH):
        if category == str(3): # 判斷是否為有效物件
            continue
        for image in os.listdir(SAMPLE_PATH+category): # 讀取樣本
            img = cv2.imread(SAMPLE_PATH+category+"/" + image)
            desc =  np.array([getMask_1(img),getMask_2(img)]).ravel().tolist() # 取得圖片皮膚色及深藍色區塊的占比做為特徵
            x += [desc]
            y += category
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print("x",x.shape)
    print("y",y.shape)
    svm.trainAuto(x, cv2.ml.ROW_SAMPLE, y) # 進行訓練
    svm.save(output_model_path) # 輸出模型

def getMask_1(img):
    upper  = np.array([105, 160,200])     #  皮膚色上限
    lowwer = np.array([60, 100,130])      #  皮膚色下限
    mask = cv2.inRange(img,lowwer, upper) #  找出皮膚色位置
    num = np.count_nonzero(mask)          #  找出皮膚色區塊大小 
    all = mask.shape[0]*mask.shape[1]     
    return num / all                      #  算出出皮膚色區塊占比
    
    
    
def getMask_2(img):
    upper  = np.array([100, 60, 50])      #  深藍色上限
    lowwer = np.array([50,  30, 10])      #  深藍色下限
    mask = cv2.inRange(img,lowwer, upper) #  找出深藍色位置
    num = np.count_nonzero(mask)          #  找出深藍色區塊大小 
    all = mask.shape[0]*mask.shape[1]     
    return num / all                      #  算出出深藍色區塊占比

def trainvalid():
    x=[]
    y=[]
    svm = cv2.ml.SVM_create()      #  ┐
    svm.setType(cv2.ml.SVM_C_SVC)  #    
    svm.setKernel(cv2.ml.SVM_RBF)  #  創建SVM模型，並設定參數  
    svm.setC(0.1)                  #    
    svm.setGamma(1.0)              #  ┘     
    for category in os.listdir(SAMPLE_PATH): 
        if int(category) == 3:  # 判斷是否為有效物件
            results = '3'
            print("invalid")
        else:
            results = '5'
            print("valid")
        for image in os.listdir(SAMPLE_PATH+category):
            img = cv2.imread(SAMPLE_PATH +category+"/" + image) # 讀取樣本
            ratio = img.shape[0]/img.shape[1]  # 計算長寬比
            x.append(ratio) # 紀錄特徵
            y += results # 紀錄標籤
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print("x",x.shape)
    print("y",y.shape)
    svm.trainAuto(x, cv2.ml.ROW_SAMPLE, y) # 進行訓練
    svm.save("valid.xml")   # 輸出模型

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
        for category in os.listdir(SAMPLE_PATH):
            if category == str(3) :
                continue
            print("===========================",str(category),"===========================")
            for image in os.listdir( SAMPLE_PATH+category):

                img = cv2.imread(SAMPLE_PATH+ category +"/" + image)
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
    train()
    trainvalid()
    
    # isValid()
    # for category in os.listdir(SAMPLE_PATH):
    #     if category == str(3):
    #         continue
    #     for image in os.listdir(SAMPLE_PATH+ "/" + category):
    #         img = cv2.imread(SAMPLE_PATH+ "/" + category+ "/" + image)
           
    #         print(category ,int(predict(img)))
            
        



