import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from global_config import Config
import tensorflow as tf
from tensorflow import keras
import tf2onnx
import onnx

config = Config()
config.training_params()

model = keras.models.load_model(config.MODEL_PATH)
model.save('tfmodel', save_format='tf')

# onnx_model = tf2onnx.convert.from_keras(model)

# onnx.save_model(onnx_model, os.path.join(config.MODEL_FOLDER, "model.onnx"))