from tensorflow.keras.models import load_model
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.python.keras.backend import set_session
import numpy as np
from PIL import Image
import sys
from collections import defaultdict, Counter
sys.path.append('..')
from global_config import Config
import os
from tqdm import tqdm

config = Config()
# config.terrain_params()
config.training_params()

# Load the model
# MODEL_NAME = config.MODEL_PATH
# VALID_PATH = config.VAL_FOLDER
MODEL_NAME = config.MODEL_PATH
VALID_PATH = config.TEST_FOLDER

dict={0:'miss',1:'in',2:'perfect'}

def classify(model, image):
    result = model.predict(image)
    themax = np.argmax(result)
    return result

def load_images_from_folder(folder_path):
    images = []
    filenames = []
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        try:
            img = Image.open(img_path)
            img = img.resize((249, 249))
            img_array = np.array(img)/255.0
            images.append(img_array)
            filenames.append(img_path)
        except:
            print("Error loading image:", img_path)
    return np.array(images), filenames

def load_image(image_fname, array=True):
    if array:
        img = Image.fromarray(image_fname)
    else:
        img = Image.open(image_fname)
    img = img.resize((249, 249))
    imgarray = np.array(img)/255.0
    final = np.expand_dims(imgarray, axis=0)
    return final

def classify_batch(model, images):
    results = model.predict(images)
    labels = [dict[np.argmax(result)] for result in results]
    probs = [np.max(result) for result in results]
    return labels, probs



def main():
    print("Loading model from", MODEL_NAME)
    model = load_model(MODEL_NAME)
    print("Done")

    print("Now classifying files in", VALID_PATH)

    for directory in os.listdir(VALID_PATH):
        folder_path = os.path.join(VALID_PATH, directory)
        if os.path.isdir(folder_path):
            print(f"\nProcessing folder: {directory}")
            images, filenames = load_images_from_folder(folder_path)
            
            if len(images) == 0:
                print(f"No valid images found in {directory}")
                continue
            
            labels, probs = classify_batch(model, images)
            
            # Print individual results
            # for filename, label, prob in zip(filenames, labels, probs):
            #     print(f"Image: {filename}, Label: {label}, Confidence: {prob:.2f}")
            
            # Print summary
            label_counts = Counter(labels)
            total_images = len(labels)
            print(f"\nSummary for {directory}:")
            print(f"Total images processed: {total_images}")
            for label, count in label_counts.items():
                percentage = (count / total_images) * 100
                print(f"{label}: {count} ({percentage:.2f}%)")
            
            majority_label = max(label_counts, key=label_counts.get)
            majority_percentage = (label_counts[majority_label] / total_images) * 100
            print(f"Majority class: {majority_label} ({majority_percentage:.2f}%)")



if __name__ == '__main__':
    main() 

