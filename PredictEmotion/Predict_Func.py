from tensorflow import keras
from tensorflow.keras.models import Model
import tensorflow as tf
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix
from PIL import Image
cnn = keras.models.load_model('final_model_2.h5')
def Predict_Mood(x):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    img = cv2.imread(x)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        faces = img[y:y + h, x:x + w]
        cv2.imwrite('face.jpg', faces)
    cv2.imwrite('detected.jpg', img)
    image=Image.open('face.jpg')
    greyscale_image = image.convert('L')
    pixels=np.asarray(greyscale_image)
    new_image = greyscale_image.resize((48,48))
    pixels=np.asarray(new_image)
    final=pixels.reshape((1,pixels.shape[0], pixels.shape[1], 1))
    final = final / 255
    prediction=cnn.predict(final)
    k = np.argmax(prediction)
    p = np.max(prediction)
    return (k,p)
