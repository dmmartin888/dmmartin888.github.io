import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt
import time

#This makes and trains the model and outputs performance metrics.
#https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.2-using-convnets-with-small-datasets.ipynb
#Modified to fit apis and bombus instead of dogs and cats, plots added.

PATH = os.path.join(os.getcwd(), 'data') #path to the train/val data

train_dir = os.path.join(PATH, 'train') #joins PATH and train and saves it to variable
validation_dir = os.path.join(PATH, 'validation') #joins PATH and validation to var

train_apis_dir = os.path.join(train_dir, 'apis') #joins train_dir to apis
train_bombus_dir = os.path.join(train_dir, 'bombus') #joins train_dir to bombus
validation_apis_dir = os.path.join(validation_dir, 'apis') #joins validation_dir to apis
validation_bombus_dir = os.path.join(validation_dir, 'bombus') #joins validation_dir to bombus

num_apis_tr = len(os.listdir(train_apis_dir)) #takes the num of images in this dir and stores it
num_bombus_tr = len(os.listdir(train_bombus_dir)) #takes the num of images in bombus dir and stores it

num_apis_val = len(os.listdir(validation_apis_dir)) #stores num of apis val images
num_bombus_val = len(os.listdir(validation_bombus_dir)) #stores num of bombus val images

total_train = num_apis_tr + num_bombus_tr #adds up bombus and apis train nums for total
total_val = num_apis_val + num_bombus_val #adds up bombus and apis val for total

print('total training apis images: ', num_apis_tr)
print('total training bombus images: ', num_bombus_tr)

print('total validation apis images: ', num_apis_val)
print('total validation bombus images: ', num_bombus_val)
print('--')
print('total training images: ', total_train)
print('total validation images: ', total_val)

batch_size = 32 #variable for batch size
epochs = 100 #variable for epochs
IMG_HEIGHT = 150 #image height var
IMG_WIDTH = 150 #image width var

validation_image_generator = ImageDataGenerator(rescale=1./255) #generator for val data

#generates training data with rotation, shifts, flips, zooms, etc to solve overfittin
image_gen = ImageDataGenerator(rescale=1./255, rotation_range=45, width_shift_range=.15, height_shift_range=.15, horizontal_flip=True, zoom_range=0.5, )
#loads images from disk, applies rescaling, and resizes images into required dimensions
train_data_gen = image_gen.flow_from_directory(batch_size=batch_size, directory=train_dir, shuffle=True, target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='binary')

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
#does same thing, but for val data
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size, directory=validation_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='binary')

sample_training_images, _ = next(train_data_gen) #next returns a batch from dataset, and the variables take only the pictures, not the labels

#function that plots images into 1 row and 5 columns
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20)) #sets the # of rows, columns, and the size of each figure
    axes = axes.flatten() #flattens axes? look into what this means
    for img, ax in zip(images_arr, axes): #for im in images_arr and ax in axes
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

#plotImages(sample_training_images[:5]) #calls the function
#plotImages(augmented_images)

#defines the model as sequential, adds all the layers
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#compiles model
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])

#outputs a summary of the layers of the model
model.summary()


#trains the model
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

#From Image Classification tutorial on Tensorflow website
#https://www.tensorflow.org/tutorials/images/classification
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

#plt.figure(figsize=(8, 8))
#plt.subplot(1, 2, 1)
#plt.plot(epochs_range, acc, label='Training Accuracy')
#plt.plot(epochs_range, val_acc, label='Validation Accuracy')
#plt.legend(loc='lower right')
#plt.title('Training and Validation Accuracy')
#plt.subplot(1, 2, 2)
#plt.plot(epochs_range, loss, label='Training Loss')
#plt.plot(epochs_range, val_loss, label='Validation Loss')
#plt.legend(loc='upper right')
#plt.title('Training and Validation Loss')
#plt.show()

#Saves the model
model_path = "./apis_bombus_model_save"
model.save(model_path)

