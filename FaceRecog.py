import os
import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

from keras.layers import Dense, GlobalAveragePooling2D

from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model

from keras.optimizers import Adam

from keras_vggface.vggface import VGGFace

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory('./AttSys/Headshots', target_size=(224,224), 
                                                    color_mode='rgb', batch_size=32, 
                                                    class_mode='categorical', shuffle=True)

train_generator.class_indices.values()
# dict_values([0, 1, 2])
NO_CLASSES = len(train_generator.class_indices.values())

base_model = VGGFace(include_top=False,
model='vgg16',
input_shape=(224, 224, 3))
base_model.summary()
print(len(base_model.layers))
# 19 layers after excluding the last few layers
x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)

# final layer with softmax activation
preds = Dense(NO_CLASSES, activation='softmax')(x)

# create a new model with the base model's original input and the 
# new model's output
model = Model(inputs = base_model.input, outputs = preds)
#model.summary()

# don't train the first 19 layers - 0..18
for layer in model.layers[:19]:
    layer.trainable = False

# train the rest of the layers - 19 onwards
for layer in model.layers[19:]:
    layer.trainable = True
model.summary()

model.compile(optimizer='Adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])
model.fit(train_generator,
  batch_size = 1,
  verbose = 1,
  epochs = 20)
# creates a HDF5 file
model.save(
    'transfer_learning_trained' +
    '_face_cnn_model.h5')
from keras.models import load_model

# deletes the existing model
del model

# returns a compiled model identical to the previous one
model = load_model(
    'transfer_learning_trained' +
    '_face_cnn_model.h5')

#assign index to each name
import pickle

class_dictionary = train_generator.class_indices
class_dictionary = {
    value:key for key, value in class_dictionary.items()
}
print(class_dictionary)
# save the class dictionary to pickle
face_label_filename = 'face-labels.pickle'
with open(face_label_filename, 'wb') as f: pickle.dump(class_dictionary, f)
