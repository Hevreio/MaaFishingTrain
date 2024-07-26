import os
# import socket

# s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# s.connect(("8.8.8.8", 80))


class Config:
    # def __init__(self):
        # self.IP = s.getsockname()[0]
        # self.RASPI_IP = "172.25.106.79" # when connected to the NUS Wifi
        # self.RASPI_IP = "192.168.202.110" # usual ip when connected to hot spot

    # def pose_params(self):
    #     # designating the ports for the pose stream
    #     self.POSE_JSON_TX_URL = self.IP
    #     self.POSE_JSON_TX_PORT = 5000
    #     self.POSE_JSON_TX_SITE = f"http://{self.POSE_JSON_TX_URL}:{self.POSE_JSON_TX_PORT}/pose_detection.json"

    #     self.POSE_MJPG_RX_URL = self.RASPI_IP
    #     self.POSE_MJPG_RX_PORT = 5000 # port from raspberry pi, pose stream
    #     self.POSE_MJPG_RX_SITE = f"http://{self.POSE_MJPG_RX_URL}:{self.POSE_MJPG_RX_PORT}/stream.mjpg"

    def training_params(self):
        # terrain image images
        # designating the folders in the classification folder
        self.BASE_PATH = os.path.abspath(os.path.dirname(__file__))
        # self.CLASSIFICATION_PATH = os.path.join(self.BASE_PATH, "terrain")
        self.CLASSIFICATION_PATH = self.BASE_PATH
        self.VIDEO_FOLDER = os.path.join(self.CLASSIFICATION_PATH, "videos")
        self.FRAME_FOLDER = os.path.join(self.CLASSIFICATION_PATH, "frames")
        self.DATASET_FOLDER = os.path.join(self.CLASSIFICATION_PATH, "dataset")
        self.MODEL_FOLDER = os.path.join(self.CLASSIFICATION_PATH, "models")

        self.TRAIN_FOLDER = os.path.join(self.DATASET_FOLDER, "train")
        self.VAL_FOLDER = os.path.join(self.DATASET_FOLDER, "val")
        self.TEST_FOLDER = os.path.join(self.DATASET_FOLDER, "test")

        # designating the paths for the terrain classification train
        self.TRAIN_SPLIT = 0.7
        self.VAL_SPLIT = 0.15
        self.TEST_SPLIT = 0.15

        # self.CLASSES = ["tile", "brick", "wood","cement","yellow buffer","gravel"]
        self.CLASSES = ["miss","in","perfect"]
        self.NUM_CLASSES = len(self.CLASSES)

        self.LEARNING_RATE = 0.001
        self.BATCH_SIZE = 64
        self.EPOCHS = 7

        self.MODEL_PATH = os.path.join(self.MODEL_FOLDER, "model.h5")
        self.ROI = (1589,232,552,543)

        # imu params
        # designating the folders in the classification folder
        # self.IMU_PATH = os.path.join(self.BASE_PATH, "imu_class")

        # self.IMU_TRAIN_FOLDER = os.path.join(self.IMU_PATH, "train")
        # self.IMU_VAL_FOLDER = os.path.join(self.IMU_PATH, "val")
        # self.IMU_TEST_FOLDER = os.path.join(self.IMU_PATH, "test")

        # self.IMU_MODEL_PATH = os.path.join(self.IMU_PATH, "imu_class.h5")

        self.WEIGHTS = 0.7

        # designating the addresses and the ports for terrain classification communication
    # def terrain_params(self):
    #     self.TERRAIN_JSON_TX_URL = self.IP
    #     self.TERRAIN_JSON_TX_PORT = 5050
    #     self.TERRAIN_JSON_TX_SITE = f"http://{self.TERRAIN_JSON_TX_URL}:{self.TERRAIN_JSON_TX_PORT}/terrain_result.json"

    #     self.TERRAIN_MJPG_RX_URL = self.RASPI_IP
    #     self.TERRAIN_MJPG_RX_PORT = 8080
    #     self.TERRAIN_MJPG_RX_SITE = f"http://{self.TERRAIN_MJPG_RX_URL}:{self.TERRAIN_MJPG_RX_PORT}/?action=stream"

    # def set_raspi_ip(self, raspi_ip):
    #     self.RASPI_IP = raspi_ip
    #     if not self.TERRAIN_MJPG_RX_URL == None:
    #         self.TERRAIN_MJPG_RX_URL = self.RASPI_IP
    #         self.TERRAIN_MJPG_RX_PORT = 8080
    #         self.TERRAIN_MJPG_RX_SITE = f"http://{self.TERRAIN_MJPG_RX_URL}:{self.TERRAIN_MJPG_RX_PORT}/stream.mjpg"
    #     if not self.POSE_MJPG_RX_URL == None:
    #         self.POSE_MJPG_RX_URL = self.RASPI_IP
    #         self.POSE_MJPG_RX_PORT = 5000 # port from raspberry pi, pose stream
    #         self.POSE_MJPG_RX_SITE = f"http://{self.POSE_MJPG_RX_URL}:{self.POSE_MJPG_RX_PORT}/stream.mjpg"














