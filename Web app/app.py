from __future__ import division, print_function
import os
import numpy as np
import cv2
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import tensorflow as tf
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
# Define a flask app
app = Flask(__name__)

# Load your trained model
loaded_model = tf.keras.models.load_model('./models/bestModel.h5')

print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img,(150,150),interpolation=cv2.INTER_NEAREST)
    cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = img/255
    img  = img.reshape(-1,150,150,1)
    preds = np.argmax(model.predict(img),axis=-1)
    return preds

@app.route('/')
@app.route('/home', methods=['GET'])
def index():
    # Main page
    return render_template('details.html')

@app.route('/data', methods=['GET', 'POST'])
def getData():
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/form' to submit form"
    if request.method == 'POST':
        form_data = request.form
        return render_template('data.html',form_data = form_data)



@app.route('/readme')
def readme():
    # README page
    return render_template('readme.html')

@app.route('/about')
def about():
    # About us page
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)        
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path, loaded_model)
        if preds[0]:
            result = "Positive"
        else:
            result = "Negative"
        
        return result
    return None



if __name__ == '__main__':
    app.run(debug=True)