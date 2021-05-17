from tensorflow import keras
from tensorflow.keras.models import Model
import tensorflow as tf
import cv2
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix
font = cv2.FONT_HERSHEY_SIMPLEX
cnn = keras.models.load_model('final_model_2.h5')
from PIL import Image
from Predict_Func import *
from Youtube import *
from g_tts import *
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    # print()
    cv2.imshow('Detecting Emotion', img)
    # os.remove('face.jpg')
    k = cv2.waitKey(30) & 0xff
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('input.png',img)
        k,p=Predict_Mood('input.png')
        print()
        print(emotions[k],p)
        s=f'Hello user! You seem to be {emotions[k]}. Let me play some songs for you.'
        TextSpeech(s)
        img = cv2.putText(img, f'Predicted Label: {emotions[k]}-{p:.4f}', (50,50), font, 
                   1, (255,0,0), 2, cv2.LINE_AA)
        break
cap.release()
cv2.imshow('Emotion Detected',img)
cv2.waitKey(100)
cv2.destroyAllWindows()
play(emotions[k])
