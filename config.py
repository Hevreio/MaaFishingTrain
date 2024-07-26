import os

BASE_PATH = "/root/SWS3009_team_astra/classification/"
VIDEO_FOLDER = os.path.join(BASE_PATH, "videos")
FRAME_FOLDER = os.path.join(BASE_PATH, "frames")
DATASET_FOLDER = os.path.join(BASE_PATH, "dataset")
MODEL_FOLDER = os.path.join(BASE_PATH, "models")

TRAIN_FOLDER = os.path.join(DATASET_FOLDER, "train")
VAL_FOLDER = os.path.join(DATASET_FOLDER, "val")
TEST_FOLDER = os.path.join(DATASET_FOLDER, "test")

TRAIN_SPLIT = 0.75
VAL_SPLIT = 0.1

CLASSES = ["tile", "brick", "wood","cement","yellow buffer"]

# default learning parameters
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
EPOCHS = 4

MODEL_PATH = os.path.join(MODEL_FOLDER, "model.h5")