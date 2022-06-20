import sys
import numpy as np
import tensorflow as tf

from tensorflow import keras

def result(input):
    img_height = 180
    img_width = 180

    model = keras.models.load_model("C:/Users/khcho/Desktop/color_model.h5")
    test_path = input
    img = keras.preprocessing.image.load_img(test_path, target_size = (img_height, img_width))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    class_names = ['black', 'blue', 'brown', 'green', 'grey', 'orange', 'pink', 'red', 'skyblue', 'violet', 'white', 'yellow']

    return(class_names[np.argmax(score)])