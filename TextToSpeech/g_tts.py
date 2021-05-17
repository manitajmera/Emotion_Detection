from gtts import gTTS
import os
from playsound import playsound
def TextSpeech(s):
    tts = gTTS(text=s,lang='en')
    filename='demo.mp3'
    tts.save(filename)
    playsound(filename)
    os.remove(filename)
# TextSpeech('Hello')