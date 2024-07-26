import sys
sys.path.append("..")
from global_config import Config
import cv2
from test import classify, load_image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.preprocessing import image
import tensorflow as tf
from PIL import Image
import json
# import argparse
import requests
from terrain.predict import classify_image


config = Config()
config.terrain_params()
config.training_params()
model = tf.keras.models.load_model(config.MODEL_PATH)

# argparser = argparse.ArgumentParser()
# argparser.add_argument("--mode", type=str, default="webcam", help="webcam or image")
# argparser.add_argument("--path", type=str, default="images", help="path to image folder")
# argparser.add_argument("--json_url", '-w',type=str, default="172.25.98.2", help="URL to send JSON data")
# argparser.add_argument("--json_port", type=int, default=5000, help="Port to send JSON data")
# args = argparser.parse_args()

def send_json(json_raw_data):
    try:
        print(json_raw_data)
        json_data = json.dumps(json_raw_data)
        response = requests.post(config.TERRAIN_JSON_TX_SITE, data=json_data, headers={'Content-Type': 'application/json'})
        print(response.text)
    except Exception as e:
        print(f"Error sending data via HTTP: {e}")

def image_inference(image):
    img = load_image(image)
    label,prob,_ = classify(model, img)
    return label, prob

def mjpg_stream_inference():
    cap = cv2.VideoCapture(config.TERRAIN_MJPG_RX_SITE)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            label, prob = image_inference(frame)
            print(f"Label: {label}, Probability: {prob}")
            json_data = {
                "label": label,
                "probability": float(prob)
            }
            send_json(json_data)
            cv2.imshow("Terrain Inference", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
        
if __name__ == "__main__":
    print("Starting MJPG stream inference...")
    mjpg_stream_inference()
