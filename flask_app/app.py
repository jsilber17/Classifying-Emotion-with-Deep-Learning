#importing libraries
import os
import numpy as np
import flask
import librosa
import matplotlib.pyplot as plt
from werkzeug import secure_filename
from flask import Flask, render_template, request
from keras.models import load_model
from librosa.display import specshow
import cv2
from PIL import Image, ImageOps
from keras import backend as K
K.clear_session()

def create_spectrogram(clip, sample_rate):
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename  = 'images/img_test2.jpeg'
    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del clip,sample_rate,fig,ax,S
    return filename

def AudioPrediction(clip, sample_rate):
    filename = create_spectrogram(clip, sample_rate)
    model = load_model('final_binary_model.h5')
    arr = cv2.imread(filename)
    res = cv2.resize(arr, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    res = res.reshape(1, 64, 64, 3)
    pred = model.predict_proba(res)
    K.clear_session()
    # top_3 = pred.argsort()[0][::-1][:3]
    # dict = {0: 'Neutral', 1: 'Calm', 2: 'Happy', 3: 'Sad', 4: 'Angry', 5: 'Fearful', 6: 'Disgust', 7: 'Surprised'}
    # top_3_classes = [dict[i] for i in top_3]
    # top_3_percent = pred[0][[top_3]]*100
    # top_3_percent_list = [round(float(i), 2) for i in top_3_percent]
    # all = (top_3_classes, top_3_percent_list)
    return pred


app=Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/', methods=['POST'])
def get_audio():
    file = request.form['audio']
    clip, sample_rate = librosa.core.load(file, sr=None, res_type='kaiser_fast')
    pred = AudioPrediction(clip, sample_rate)
    return render_template('results.html', data=pred)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
