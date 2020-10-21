# https://www.pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python/
# prerequisite :
# pip install "picamera[array]"

from picamera.array import PiRGBArray  # Generates a 3D RGB array
from picamera import PiCamera  # Provides a Python interface for the RPi Camera Module
import time  # Provides time-related functions
import numpy as np
import cv2
import tensorflow as tf

# Open tflite model
interpreter = tf.lite.Interpreter(model_path="tflite_models/converted_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# Get input shape
input_shape = input_details[0]['shape']

# Initialize the camera
camera = PiCamera()

# Set the camera resolution
camera.resolution = (640, 480)

# Set the number of frames per second
camera.framerate = 32

# Generates a 3D RGB array and stores it in rawCapture
raw_capture = PiRGBArray(camera, size=(640, 480))

# Wait a certain number of seconds to allow the camera time to warmup
time.sleep(0.1)

# Labels
labels = ['cans', 'oranges', 'plastic']

# Capture frames continuously from the camera
for frame in camera.capture_continuous(raw_capture, format="rgb", use_video_port=True):

    # Grab the raw NumPy array representing the image
    image = frame.array
    # Reshape input_image
    np.reshape((image, input_shape))

    # Casting to uint8
    info = np.iinfo(image.dtype)  # Get the information of the incoming image type
    image = image.astype(np.float64) / info.max  # normalize the data to 0 - 1
    image = 255 * image  # Now scale by 255
    image = image.astype(np.float32)

    # Set the value of Input tensor
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    # prediction for input data
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Get the label class
    label = labels[np.argmax(output_data)]
    # Score
    score = str(round(np.max(output_data), 2))
    # Label to print
    label2print = label+": "+score

    # Display the resulting frame
    cv2.putText(frame, label2print, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
    # Display the frame using OpenCV
    cv2.imshow("Frame", image)

    # Wait for keyPress for 1 millisecond
    key = cv2.waitKey(1) & 0xFF

    # Clear the stream in preparation for the next frame
    raw_capture.truncate(0)

    # If the `q` key was pressed, break from the loop
    if key == ord("q"):
        break




