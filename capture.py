import os
import cv2
import time
from datetime import datetime
import sys
sys.path.append('..')
from global_config import Config
config = Config()
config.terrain_params()
config.training_params()

date_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
# save_dir = os.path.join(os.getcwd(), "frames", date_dir)
save_dir = os.path.join(config.FRAME_FOLDER, date_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# stream_url = "http://192.168.202.110:5000//stream.mjpg"
stream_url = config.TERRAIN_MJPG_RX_SITE
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()  

frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        frame_count += 1
        frame_path = os.path.join(save_dir,  f"{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)
        time.sleep(1. / 15)


except KeyboardInterrupt:
    print("Keyboard interrupt.")

finally:
    cap.release()
    cv2.destroyAllWindows()


