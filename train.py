import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import config
import sys 
# sys.path.append('/root/SWS3009_team_astra/')
from global_config import Config
from tensorflow.keras.applications.resnet_v2 import ResNet101V2, ResNet50V2, ResNet152V2
from tensorflow.keras.applications.efficientnet import EfficientNetB2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD


config = Config()
# config.terrain_params()
config.training_params()

MODEL_PATH = config.MODEL_PATH
# TRAIN_DIR = os.path.join(config.DATASET_FOLDER, 'train')
TRAIN_DIR = config.TRAIN_FOLDER
# VAL_DIR = os.path.join(config.DATASET_FOLDER, 'val')
VAL_DIR = config.VAL_FOLDER
BATCH_SIZE = config.BATCH_SIZE

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

# create the base pre-trained model
def create_model(num_hidden, num_classes):

    base_model = ResNet50V2(weights='imagenet', include_top=False)
    # base_model = EfficientNetB2(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(num_hidden, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    for layer in base_model.layers:
        layer.trainable = False
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    return model

def load_existing_model():
    model = load_model(MODEL_PATH)
    
    numlayers = len(model.layers)

    for layer in model.layers[:numlayers-3]:
        layer.trainable = False

    for layer in model.layers[numlayers-3:]:
        layer.trainable = True

    return model

def train(model_file, train_path, validation_path, num_hidden=200, num_classes=config.NUM_CLASSES, steps=32, num_epochs=config.EPOCHS):

    if os.path.exists(model_file):
        print('There is an existing model')
        model = create_model(num_hidden, num_classes)
    else:
        print('Creating new model')
        model = create_model(num_hidden, num_classes)

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    checkpoint = ModelCheckpoint(model_file)

    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        # shear_range=0.2,
        # zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=180,
        fill_mode='nearest'
        )

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(\
        train_path,\
        target_size=(224, 224),\
        batch_size=BATCH_SIZE,\
        class_mode='categorical',
        classes=config.CLASSES)
    

    validation_generator = test_datagen.flow_from_directory(\
        validation_path,\
        target_size=(249, 249),
        batch_size=6,
        class_mode='categorical',
        classes=config.CLASSES)


    model.fit(
        train_generator,
        steps_per_epoch=None,
        epochs=num_epochs,
        validation_data=validation_generator,
        validation_steps=5,
        callbacks=[checkpoint]
    )

    for layer in model.layers[:-2]:
        layer.trainable = False

    for layer in model.layers[-2:]:
        layer.trainable = True

    model.compile(optimizer=SGD(learning_rate=0.00001, momentum=0.9), loss='categorical_crossentropy')

    model.fit(
        train_generator,
        steps_per_epoch=steps,
        epochs=num_epochs - 3 ,
        callbacks=[checkpoint],
        validation_data=validation_generator,
        validation_steps=10
    )

def main():
    train(MODEL_PATH, TRAIN_DIR, VAL_DIR)

if __name__ == '__main__':
    main()