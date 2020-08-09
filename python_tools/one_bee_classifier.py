import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.preprocessing import image

#Used to evaluate a single image and get probability prediction from it.
#Modified for bees and saving and loading.
#https://www.tensorflow.org/tutorials/images/classification
labels = ['apis', 'bombus'] #sets up labels

im_loc = '/home/daphne/Desktop/data/validation/bombus/4068.jpg' #sets up image location

#formats image to size
im_PIL_format = tf.keras.preprocessing.image.load_img(im_loc, color_mode='rgb', target_size=(150,150), interpolation='nearest')

#switches PIL Image to numpy array
im_array = tf.keras.preprocessing.image.img_to_array(im_PIL_format) / 255
#formats it into a form that the model will actually take
im = (np.expand_dims(im_array, 0))

#loads and makes a prediction, finds correct label based on prediction
probability_model = tf.keras.models.load_model("./bee_model_save_rgb.h5/saved_model.h5")
predictions = probability_model.predict(im)
print("&&&&&&&&&&&&")
print(predictions)

if predictions[0][0] <= 0.5:
    im_label = labels[0]

else:
    im_label = labels[1]

#shows the image with the label
plt.imshow(im_PIL_format)
plt.title(im_label)
plt.show()

