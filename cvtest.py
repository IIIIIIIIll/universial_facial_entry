import cv2 as cv
import os
from model_trainer import Model


TAKE_SAMPLES = False # True if you are building your face sample
cap = cv.VideoCapture(0)
model = Model()
model.load()
blue = (102,255,255)
count = 0
CropPadding = 50
dir_path = "/home/yut/ML_CV/keras_learn/Webcam/target"

def extendFaceRect(rect):
    [x, y, w, h] = rect
    if y > CropPadding: y = y - CropPadding
    else: y = 0
    h += 2*CropPadding
    if x > CropPadding: x = x - CropPadding
    else: x = 0
    w += 2*CropPadding
    return [x, y, w, h]


while True:
    _, frame = cap.read()

    grey = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    cropped = grey
    cascade = cv.CascadeClassifier(cv.haarcascades+"haarcascade_frontalface_default.xml")
    faceret = cascade.detectMultiScale(grey,scaleFactor=1.2,minNeighbors=3,minSize=(85,85))
    # print(frame.shape)
    # print(faceret)
    if len(faceret)>0:
        for rect in faceret:
            [x,y,width,height] = extendFaceRect(rect)
            cv.rectangle(frame, (rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]), blue,2)
            cropped = grey[y:y+height,x:x+width]
            if(TAKE_SAMPLES):
                outfile = count.__str__() + '.jpg'
                path = os.path.join(dir_path,outfile)
                cv.imwrite(path,cropped)
            result = model.predict(cropped,img_channels=1)
            if result == 0:
                print("Targeted",count )
                count += 1
            else:
                print("NONE",count)
                count += 1

    cv.imshow('frame',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
