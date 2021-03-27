import numpy as np
import cv2 
import cv2 as cv



# face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')

img = cv.imread('IMG-20210323-WA0037.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cropped_face = img[y:y+h+50, x:x+w+50]
    # cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    x=x-10
    y=y-10
   
cv2.imwrite('3.jpg', cropped_face)

cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()