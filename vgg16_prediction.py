# Import required libraries
from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import pickle

# Set dataset paths
TrainingImagePath = '/content/drive/MyDrive/preprocessed dataset/train-skin'
TestingImagePath = '/content/drive/MyDrive/preprocessed dataset/test-skin'

# Data Augmentation
Train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=180,
    vertical_flip=True,
    horizontal_flip=True,
    brightness_range=[0.5, 1.5],
    zoom_range=[1, 1.5]
)

Test_datagen = ImageDataGenerator(rescale=1./255)

# Load training and testing dataset
training_set = Train_datagen.flow_from_directory(
    TrainingImagePath,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)

testing_set = Test_datagen.flow_from_directory(
    TestingImagePath,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)

# Print class indices
print("Training samples:", training_set.samples)
print("Testing samples:", testing_set.samples)

# Mapping disease names
TrainClasses = training_set.class_indices
ResultMap = {value: key for key, value in TrainClasses.items()}



OutputNeurons = len(ResultMap)
print("Number of output neurons:", OutputNeurons)

# **Load VGG16 Model (Pretrained on ImageNet)**
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze all layers except the last few
for layer in base_model.layers[:-4]:
    layer.trainable = False  # Keep pretrained layers frozen

# Add custom classification layers on top of VGG16
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)  # Dropout to prevent overfitting
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(OutputNeurons, activation='softmax')(x)  # Output layer

# Define final model
model = Model(inputs=base_model.input, outputs=x)

# Compile Model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# Model Summary
model.summary()

# Train the Model
model_history = model.fit(
    training_set,
    steps_per_epoch=len(training_set),
    epochs=40,  # Less epochs since VGG16 is already trained
    validation_data=testing_set,
    validation_steps=len(testing_set),
    verbose=1
)

# Save Model
model.save("/content/drive/MyDrive/preprocessed dataset/updated_vgg16_skin_disease.h5")

# Save Model using Pickle
#pickle.dump(model, open('/content/drive/MyDrive/MyDrive/preprocessed dataset/new_model_skin_vgg.pkl', 'wb'))