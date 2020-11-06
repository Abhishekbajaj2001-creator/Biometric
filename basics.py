import face_recognition
import numpy as np
import cv2

imgelon=face_recognition.load_image_file('images/elon.jpg')
imgelon=cv2.cvtColor(imgelon,cv2.COLOR_BGR2RGB)
imgelontest=face_recognition.load_image_file('images/elon2.jpg')
imgelontest=cv2.cvtColor(imgelontest,cv2.COLOR_BGR2RGB)

faceloc=face_recognition.face_locations(imgelon)[0]
encodeelon=face_recognition.face_encodings(imgelon)[0]
cv2.rectangle(imgelon,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

faceloctest=face_recognition.face_locations(imgelontest)[0]
encodeelontest=face_recognition.face_encodings(imgelontest)[0]
cv2.rectangle(imgelontest,(faceloctest[3],faceloctest[0]),(faceloctest[1],faceloctest[2]),(255,0,255),2)

results=face_recognition.compare_faces([encodeelon],encodeelontest)
facedist=face_recognition.face_distance([encodeelon],encodeelontest)
print(results,facedist)
cv2.putText(imgelontest,f'{results} {round(facedist[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Elon musk',imgelon)
cv2.imshow('Elon test',imgelontest)
cv2.waitKey(0)


