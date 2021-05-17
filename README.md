# Emotion Detection
  
***

## Aim & Ideation

The aim of this project was learning the facial expression from an image using a CNN(convolutional neural network) model. The emotion detected was further used to play songs for the user according to the mood. 

## Overall Pipeline of the Project

![overall pipeline](images/Pipeline.png)

### Face Detection

I used OpenCV for Face Detection from images. I have implemented it for both live detection as well as from input images.

### Emotion Detection

I used FER-2013 dataset to train my emotion detection model.</br>
I built a 13 layered Convolutional Neural Network (CNN) in Keras which was able to detect 7 moods for upto 75% accuracy.</br>


### Text to speech conversion

I used the *google-text-to-speech (gtts)* API for the conversion of text responses to speech.</br>
The API uses *playsound* to play a temporary mp3 file created from the model's textual response.

***

## Usage

Install the required dependencies :

```bash
$pip install -r requirements.txt
```

Trained model can be found [here](https://drive.google.com/drive/folders/1lpbKN6hwnpNKFKxvLCBdnySoGpbku7u6?usp=sharing)

Required File Structure:

```txt
Tp_Project
├── Dataset
│   ├── test.csv
│   ├── train.csv
│   ├── icml_face_data.csv
├── final_model_2.h5
└── ...
```

### Functionality

* Facial Recognition & Mood Detection
* Plays song according to the mood detected

***

### Demonstration

The video demonstration of this project can be found [here](https://drive.google.com/drive/folders/1_WaVgxhV1kQqCMV87a6t7xllnd1TqK8C?usp=sharing).

***

## References

1. _Emotion Detection using Image Processing in Python _
   * **Link** : [https://arxiv.org/abs/2012.00659]
   * **Authors** : Raghav Puri, Archit Gupta, Manas Sikri, Mohit Tiwari, Nitish Pathak, Shivendra Goel
   * **Tags** : Computer Vision and Pattern Recognition
   * **Published** : 1 Dec, 2020

2. _Facial Expression Recognition_
   * **Link** : [https://www.kaggle.com/drbeanesp21/aliaj-final-facial-expression-recognition]
   * **Authors** : Aliaj, Marsela
   * **Published** : 4 May 2021

2. _FER-2013_ (Dataset)
   * **Link** : [https://datarepository.wolframcloud.com/resources/FER-2013]
