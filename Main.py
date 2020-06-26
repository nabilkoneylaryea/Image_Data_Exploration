# allows us to actually see the image
import matplotlib.pyplot as plt

# ACCESSING CIFAR-10 DATASET

# calls on file cifar10.py where we can access funcitons that will load batches
from keras.datasets import cifar10

# Puts dataset into two tuples or arrays...idk...are they dictionaries cause key value pairs??
# cifar10.load_data() method loads entire dataset
# every variable name represents and array
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()


# EXPLORE CIFAR-10 DATASET

# printing an image returns an array of arrays (a natrix format):
# each array of three represents a specific pixel's RGB values
# print(train_images[0])

# each array that contains all of these smaller arrays of three represent a row of pixels
# can access the rows by adding another set of brackets after position of the image with the selected row
# if a picture is 32x32 then there should be 32 rows printed out
first_image = train_images[0]
print(first_image[0])

# will show image as a graph of pixels
plt.imshow(first_image)
plt.show()
