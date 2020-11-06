import face_recognition
import numpy as np
import cv2
import os
from datetime import datetime

path='images'
imagelist =[]
imagenames=[]
mylist=os.listdir(path)

for cls in mylist:
    curimage=cv2.imread(f'{path}/{cls}')
    imagelist.append(curimage)
    imagenames.append(os.path.splitext(cls)[0])

def findencoding(imagelist):
    encodelist=[]
    for img in imagelist:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist


def markattendance(name):
    with open('attendance.csv','r+') as f:
        mydatalist = f.readline()
        namelist = []
        for line in mydatalist:
            entry=line.split(',')
            namelist.append(entry[0])
        if name not in namelist:

            ow = datetime.now()
            stringname =ow.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{stringname}')






encodelistknow=findencoding(imagelist)

cap = cv2.VideoCapture(0)

while True:
    success,img = cap.read()
    imgS = cv2.resize(img,(0,0),None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    facecurr = face_recognition.face_locations(imgS)
    encodecurr = face_recognition.face_encodings(imgS,facecurr)

    for encodeface,faceloc in zip( encodecurr, facecurr ):
        matches = face_recognition.compare_faces(encodelistknow, encodeface)
        facedist = face_recognition.face_distance(encodelistknow, encodeface)
        print(facedist)
        matchindex = np.argmin(facedist)

        if matches[matchindex]:
            name = imagenames[matchindex].upper()
            y1,x2,y2,x1= faceloc
            y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markattendance(name)

    cv2.imshow('webcam',img)
    cv2.waitKey(1)





