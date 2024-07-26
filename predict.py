import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# import config
from keras.preprocessing import image
import cv2
import os
from PIL import Image
import sys
sys.path.append('..')
sys.path.append('.')
from global_config import Config

config = Config()
config.terrain_params()
config.training_params()

model = tf.keras.models.load_model('terrain/models/model.h5')
class_names = config.CLASSES
IMG_SIZE = (224, 224)

def classify_image(model, img_path, class_names):
    processed_img = preprocess_image(img_path)
    if processed_img is None:
        return "Failed to process image"
    predictions = model.predict(processed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    return class_names[predicted_class_index]

def preprocess_image(img_path, target_size=(224, 224)):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image: {img_path}")
        return None
    
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize to [0,1]
    return img_array

test_dir = './terrain/dataset/test'
test_dir = config.TEST_FOLDER
output_dir = './labelled_images'
os.makedirs(output_dir, exist_ok=True)

for image_directory in os.listdir(test_dir):
    image_count = 0
    for img in os.listdir(os.path.join(test_dir,image_directory)):
        if image_count >= 20:
            break

        img_path = os.path.join(test_dir, image_directory, img)
        predicted_class = classify_image(model, img_path, class_names)
        print(f"Predicted class for {img}: {predicted_class}")

        img = cv2.imread(img_path)
        
        # Convert BGR to RGB (OpenCV uses BGR by default)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create a figure and axis
        fig, ax = plt.subplots()
        
        # Display the image
        ax.imshow(img_rgb)
        
        # Set the title (predicted class)
        ax.set_title(predicted_class)
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Save the figure as PNG (a supported format)
        output_filename = f"{image_directory}_{os.path.splitext(image_directory)[0]}.png"
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, format='png')
        plt.close(fig)  # Close the figure to free up memory

        print(f"Saved as PNG: {output_path}")
        image_count += 1

print(f"Labelled images saved in {output_dir}")