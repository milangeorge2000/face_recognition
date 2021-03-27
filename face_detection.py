import numpy as np
import cv2 
import cv2 as cv
from glob import glob
import os
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
count = 0
for root, dirs, files in os.walk("./sidharth"):
    for filename in files:
        count = count + 1 



        img = cv.imread(f"sidharth/{filename}")
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#  
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
          cropped_face = img[y:y+h+50, x:x+w+50]
#         # cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
          x=x-10
          y=y-10
   
        cv2.imwrite(f'cropped_sidharth/{count}.jpg', cropped_face)

