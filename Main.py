import matplotlib
#Uses a non-GUI backend to save file because having trouble using plt.show()
matplotlib.use('Agg')
# allows us to actually see the image
import matplotlib.pyplot as plt

# ACCESSING CIFAR-10 DATASET ######################################################################################

import numpy as np
import keras.utils as utils

# calls on file cifar10.py where we can access funcitons that will load batches
from keras.datasets import cifar10

# Puts dataset into two tuples or arrays...idk...are they dictionaries cause key value pairs??
# cifar10.load_data() method loads entire dataset
# every variable name represents and array
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()


# EXPLORE CIFAR-10 DATASET #######################################################################################

# printing an image returns an array of arrays (a natrix format):
# each array of three represents a specific pixel's RGB values
# print(train_images[0])

# each array that contains all of these smaller arrays of three represent a row of pixels
# can access the rows by adding another set of brackets after position of the image with the selected row
# if a picture is 32x32 then there should be 32 rows printed out

# CAN VIEW IMAGE DATA LIKE THIS ##################################################
# fig_num = 0
# first_image = train_images[fig_num]
# print(first_image[fig_num])

# WILL SHOW IMAGE IN GRAPH OF PIXELS
#
# plt.imshow(first_image)
# fig_name = 'matplot-figure-' + (fig_num + 1)
# try:
#     plt.savefig(fig_name)
# except:
#     print('Img already on file.')

# HOW TO PROPERLY SHOW LABEL ##################################################
#
# labels are represented as an index of an array
# for this dataset you can make an array of strigs to represent the actual labels
labels_array = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# ONE WAY TO DO IT
# need two sets of brackets because calling an index of train_labels returns an array
# first_label_index = train_labels[0][0]
# print(labels_array[first_label_index])

# THE WAY WE DO IT FOR REAL!! (CHANGING LABELS INTO CATEGORICAL DATA)
# will represent each train_labels[x] as an array of length 10 with a 1 in the correlating index position and
# 0s in the other positions
# in formatting our labels like this the computer can represent the probability of each each label easier
# called on hot encoding
# looks like this [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
# works well with softmax activation function
train_labels = utils.to_categorical(train_labels)
# HOW TO GET LABEL
# max_index = np.argmax(train_labels[0])
# print(labels_array[max_index])
test_labels = utils.to_categorical(test_labels)

# FORMATTING INPUT IMAGES (NORMALIZATION) #############################################################################
#
# instead of having large numbers (0-255) we are going to scale it down to (0-1)
# Why? Because machine doesn't like large numbers
# How To:
#   1. Convert to float
#   2. divide by 255

train_images = train_images.astype('float32')
train_images /= 255.0

test_images = test_images.astype('float32')
test_images /= 255.0

# TURNING MATRIXES INTO FLATTENED ARRAYS ##############################################################################
#
# Why? If you want to export the model into an application like an Android app ...
# ... it's easier to work with flattened arrays
# will take in an array of images e.g. test_images
def reshape_image(input_image_arrays):
    output_array = []
    for image_array in input_image_arrays:
        # flattening each image it finds in input array
        # .reshape() is kool because it preserves the order in which the values in the array appear
        output_array.append(image_array.reshape(-1))
    #returning output as numpy array because machine learning models like working with these types of arrays
    return np.asarray(output_array)