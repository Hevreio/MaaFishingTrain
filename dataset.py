import cv2
import os
from PIL import Image
import random
import shutil
# import config
import argparse
from imutils import paths
import numpy as np 
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
import sys 
sys.path.append('/root/SWS3009_team_astra/')
from global_config import Config

config = Config()
config.training_params()
config.terrain_params()


def extract_frames(video_path, output_folder, interval=3):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    video = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_count += 1
        frame_path = os.path.join(output_folder, f"{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)

        frame_count += 1

    video.release()


def delete_low_entropy_frames(image_folder=config.FRAME_FOLDER, threshold=2.5):
    images = []
    filenames = []

    for class_name in os.listdir(image_folder):
        for image_name in os.listdir(os.path.join(image_folder, class_name)):
            image_path = os.path.join(image_folder, class_name, image_name)
            if os.path.isfile(image_path):
                images.append(cv2.imread(image_path))
                filenames.append(image_path)
                print(f"Loaded {image_path}")

    num_images = len(images)
    to_delete = set()

    for i in range(num_images):
        img_gray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        img_entropy = entropy(img_gray, disk(5))
        print(f"Entropy: {img_entropy}")
        if img_entropy < threshold:
            to_delete.add(filenames[i])
            print(f"Deleting {filenames[i]}")
    for filepath in to_delete:
        os.remove(filepath)
        print(f"Deleted {filepath}")

def extract_frames_from_folder(video_folder=config.VIDEO_FOLDER, output_folder=config.FRAME_FOLDER, interval=3):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

    for video_name in os.listdir(video_folder):
        video_path = os.path.join(video_folder, video_name)
        output_video_folder = os.path.join(output_folder, video_name)
        extract_frames(video_path, output_video_folder, interval)

def build_dataset(image_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # extract_frames_from_folder(config.VIDEO_FOLDER, config.FRAME_FOLDER)

    for class_name in os.listdir(image_folder):
        imagePath = list(paths.list_images(config.FRAME_FOLDER + "/" + class_name))
        random.seed(42)
        random.shuffle(imagePath)

        # split the data into training and testing splits using default 75% training and 25% testing
        i = int(len(imagePath) * config.TRAIN_SPLIT)
        trainPaths = imagePath[:i]
        testPaths = imagePath[i:]

        # we'll use part of the training data for validation
        i = int(len(trainPaths) * config.VAL_SPLIT)
        valPaths = trainPaths[:i]
        trainPaths = trainPaths[i:]

        # define the datasets that we'll be building
        datasets = [
            ("train", trainPaths, config.TRAIN_FOLDER),
            ("val", valPaths, config.VAL_FOLDER),
            ("test", testPaths, config.TEST_FOLDER)
        ]

        for (dType, imagePaths, baseOutput) in datasets:
            # show which data split we are creating
            print(f"Building {dType} split")

            # if the output base output directory does not exist, create it
            if not os.path.exists(baseOutput):
                print(f"Creating {baseOutput} directory")
                os.makedirs(baseOutput)

            # loop over the input image paths
            for inputPath in imagePaths:
                # extract the filename of the input image along with its corresponding class label
                filename = inputPath.split(os.path.sep)[-1]
                label = inputPath.split(os.path.sep)[-2]

                # build the path to the label directory
                labelPath = os.path.sep.join([baseOutput, label])

                # if the label output directory does not exist, create it
                if not os.path.exists(labelPath):
                    print(f"Creating {label} directory")
                    os.makedirs(labelPath)

                # construct the path to the destination image and then copy the image itself
                p = os.path.sep.join([labelPath, filename])
                shutil.copy2(inputPath, p)

def delete_similar_frames(image_folder, threshold=0.9):
    
    images = []
    filenames = []

    for class_name in os.listdir(image_folder):
        for image_name in os.listdir(os.path.join(image_folder, class_name)):
            image_path = os.path.join(image_folder, class_name, image_name)
            # if os.path.isfile(image_path):
            images.append(cv2.imread(image_path))
            filenames.append(image_path)
            print(f"Loaded {image_path}")
            
            num_images = len(images)
            to_delete = set()

            
            for i in range(num_images):
                # print(f"Comparing {filenames[i]}")
                for j in range(i + 1, num_images):
                    # print(f"Comparing {filenames[j]}\n")
                    if filenames[j] in to_delete:
                        continue

                    # img1_gray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
                    # img2_gray = cv2.cvtColor(images[j], cv2.COLOR_BGR2GRAY)
                    # img1_gray = np.sqeeze(images[i])
                    # img2_gray = np.sqeeze(images[j])

                    # win_size = min(img1_gray.shape[0], img2_gray.shape[0],7)
                   
                    score = ssim(images[i], images[j], channel_axis=2)
                    print(f"SSIM score: {score}")
                    if score > threshold:
                        to_delete.add(filenames[j])
                        print(f"Deleting {filenames[j]}")
                for filepath in to_delete:
                    os.remove(filepath)
                    print(f"Deleted {filepath}")


                    

def load_dataset():
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_FOLDER,
        target_size=(224, 224),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        config.VAL_FOLDER,
        target_size=(224, 224),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        config.TEST_FOLDER,
        target_size=(224, 224),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical'
    )

    return train_generator, val_generator, test_generator
    
# https://pyimagesearch.com/2020/04/27/fine-tuning-resnet-with-keras-tensorflow-and-deep-learning/

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract_frames','-e',type=bool, default=False)
    parser.add_argument('--build_dataset','-b', type=bool, default=False)
    parser.add_argument('--delete_similar_frames','-d', type=bool, default=False)


    args = parser.parse_args()
    if args.extract_frames:
        extract_frames_from_folder()
    elif args.build_dataset:
        build_dataset(config.FRAME_FOLDER, config.DATASET_FOLDER)
    elif args.delete_similar_frames:
        delete_similar_frames(config.FRAME_FOLDER)
    
    