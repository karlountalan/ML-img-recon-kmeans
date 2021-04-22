from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename

import numpy as np
from matplotlib import image as imm
from sklearn.utils import shuffle
from sklearn.datasets import load_sample_image
from sklearn.cluster import KMeans
from PIL import Image
import os
import PIL

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    number = request.form['number']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img = Image.open(file)
        img = np.array(img,dtype=np.float64)/255
        w,h,d = original_shape = tuple(img.shape)
        print(original_shape)
        image_array = np.reshape(img,(w*h,d))

        image_sample = shuffle(image_array,random_state=42)[:1000]

        n_colors = int(number)
        print(n_colors)
        kmeans = KMeans(n_colors, random_state=42).fit(image_sample)

        labels = kmeans.predict(image_array)

        def reconstruct_image(cluster_centers,labels,w,h):
          d = cluster_centers.shape[1]
          image = np.zeros((w,h,d))
          label_index=0
          for i in range(w):
            for j in range(h):
              image[i][j] = cluster_centers[labels[label_index]]
              label_index+=1
          return image

        file_ext = "."+filename.split('.')[-1]
        file_name = filename.split('.')[0]

        pic = reconstruct_image(kmeans.cluster_centers_,labels,w,h)
        imm.imsave(str(os.path.join(app.config['UPLOAD_FOLDER']))+'/'+str(file_name)+'_out2'+file_ext,pic)
        filename2=str(file_name)+'_out2'+file_ext

        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        return render_template('index.html', filename=filename, filename2= filename2, n=n_colors)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

if __name__ == "__main__":
    app.run()
