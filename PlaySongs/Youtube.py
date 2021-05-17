import vlc
from time import sleep
import pafy
import keyboard
def play(mood):
    if mood=='Happy' or mood=='Neutral':
        url = 'https://www.youtube.com/watch?v=dXLLzpZO_YA'
    elif mood=='Surprise' or mood=='Fear' :
        url='https://www.youtube.com/watch?v=26nsBfLXwSQ'
    elif mood=='Sad' or mood=='Disgust' or mood=='Angry':
        url='https://www.youtube.com/watch?v=NqTDoPAMDGs'
    video = pafy.new(url)
    best = video.streams[0]
    media = vlc.MediaPlayer(best.url)
    media.play()
    sleep(3) # Or however long you expect it to take to open vlc
    while True:
        if keyboard.is_pressed('p'): 
            media.pause()
        elif keyboard.is_pressed('q'):
            media.stop()
            break
# play(1)