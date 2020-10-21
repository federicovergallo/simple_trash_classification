import numpy as np
import cv2
import tensorflow as tf
import os

gpus= tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

filename = 'video.avi'
frames_per_second = 10.0
res = '720p'

# Standard Video Dimensions Sizes
STD_DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}
# Video Encoding, might require additional installs
# Types of Codes: http://www.fourcc.org/codecs.php
VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    #'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}


# Set resolution for the video capture
# Function adapted from https://kirr.co/0l6qmh
def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)


# grab resolution dimensions and set video capture to it.
def get_dims(cap, res='1080p'):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width,height = STD_DIMENSIONS[res]
    ## change the current caputre device
    ## to the resulting resolution
    change_res(cap, width, height)
    return width, height


def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
      return  VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']


# Loading model
IMG_HEIGHT, IMG_WIDTH = 224, 224
model_path = "checkpoints/mobilenet_aug_lite/Epoch_275_model.hp5"
model = tf.keras.models.load_model(model_path)

# Labels
labels = ['cans', 'oranges', 'plastic']

#Get frame
cap = cv2.VideoCapture(0)
out = cv2.VideoWriter(filename, get_video_type(filename), 25, get_dims(cap, res))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Resize and reshape frame
    img = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
    img = np.expand_dims(img, axis=0)
    # Predict
    pred = model.predict(img)
    # Get the label class
    label = labels[np.argmax(pred)]
    # Score
    score = str(round(np.max(pred),2))
    # Label to print
    label2print = label+": "+score

    # Display the resulting frame
    cv2.putText(frame,label2print, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 255),2,cv2.LINE_4)
    cv2.imshow('', frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()