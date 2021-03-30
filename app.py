from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import numpy as np
import os
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Define a flask app
app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')








@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        import os, shutil
        folder = r'uploads'
        for filename in os.listdir(folder):
          file_path = os.path.join(folder, filename)
          if os.path.isfile(file_path) or os.path.islink(file_path):
              os.unlink(file_path)
          elif os.path.isdir(file_path):
             shutil.rmtree(file_path)
           
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        


        img = cv2.imread(file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
                 cropped_face = img[y:y+h, x:x+w]
                 print(cropped_face.shape)
        import os, shutil
        folder = r'cropped_face'
        for filename in os.listdir(folder):
          file_path = os.path.join(folder, filename)
          if os.path.isfile(file_path) or os.path.islink(file_path):
              os.unlink(file_path)
          elif os.path.isdir(file_path):
             shutil.rmtree(file_path)
        cv2.imwrite("cropped_face/1.jpg",cropped_face)

        img = image.load_img("cropped_face/1.jpg",target_size=(224,224)) ##loading the image
        img = np.asarray(img) ##converting to an array
        img = img / 255 ##scaling by doing a division of 255
        img = np.expand_dims(img, axis=0) ##expanding the dimensions
  
        saved_model = load_model("face_inception.h5") ##loading the model
        mask_model = load_model("mobilenet.h5")
        mask_output = mask_model.predict(img)
        mask_output = np.argmax(mask_output,axis=1)
        output = saved_model.predict(img)
        output = np.argmax(output, axis=1)
        if mask_output[0] == 1:
            
              
              if output[0] == 0:
                  result = "Arshith has No Mask"
              else:
                  result = "Sidharth has No Mask"

         ##Taking the index of the maximum value
        else:
             if output[0] == 0:
                 result = "Arshith"
             else:
                 result = "Sidharth"
  
       
        return result

if __name__ == '__main__':
    app.run(debug=False)