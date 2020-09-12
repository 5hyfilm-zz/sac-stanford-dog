import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
import numpy as np
import argparse
import imutils
import time
import uuid
import base64

IMG_WIDTH, IMG_HEIGHT = 224, 224
model_path = 'model/standford_sac_dog_model.h5'
model = load_model(model_path)

UPLOAD_FOLDER = 'upload_folder'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'JPG', 'JPEG', 'png'])

app = Flask(__name__)

def get_as_base64(url):
    return base64.b64encode(requests.get(url).content)

def predict(file):
    x = load_img(file, target_size=(IMG_WIDTH, IMG_HEIGHT))
    x = img_to_array(x)
    x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)
    array = model.predict(x)
    result = array[0]
    answer = np.argmax(result)

    if answer == 0:
        print('Label: Beagle')
    elif answer == 1:   
        print('Label: Chihuahua')
    elif answer == 2:
        print('Label: French Bulldog')
    elif answer == 3:
        print('Label: Golden Retriever')
    elif answer == 4:
        print('Label: Pomeranian')
    elif answer == 5:
        print('Label: Pug')
    elif answer == 6:
        print('Label: Rottweiler')
    elif answer == 7:
        print('Label: Shih-Tzu')
    elif answer == 8:
        print('Label: Siberian Husky')
    elif answer == 9:
        print('Label: Yorkshire Terrier')
        
    return answer

def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def template_test():
    return render_template('index.html', label='')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        import time
        start_time = time.time()
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            result = predict(file_path)
            if result == 0:
                label = 'Beagle'
            elif result == 1:
                label =	'Chihuahua'
            elif result == 2:
                label = 'French Bulldog'
            elif result == 3:
                label = 'Golden Retriever'
            elif result == 4:
                label = 'Pomeranian'
            elif result == 5:
                label = 'Pug'
            elif result == 6:
                label = 'Rottweiler'
            elif result == 7:
                label = 'Shih-Tzu'
            elif result == 8:
                label = 'Siberian Husky'
            elif result == 9:
                label = 'Yorkshire Terrier'
            
            print(result)
            print(file_path)
            filename = my_random_string(6) + filename

            os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("--- %s seconds ---" % str (time.time() - start_time))
            return render_template('index.html', label=label, imagesource='upload_folder/' + filename)

from flask import send_from_directory

@app.route('/upload_folder/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

from werkzeug import SharedDataMiddleware
app.add_url_rule('/upload_folder/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/upload_folder':  app.config['UPLOAD_FOLDER']
})

if __name__ == "__main__":
    app.debug=False
    app.run(host='localhost', port=5000)