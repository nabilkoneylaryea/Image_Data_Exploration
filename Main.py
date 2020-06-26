import matplotlib
#Uses a non-GUI backend to save file because having trouble using plt.show()
matplotlib.use('Agg')
# allows us to actually see the image
import matplotlib.pyplot as plt

# ACCESSING CIFAR-10 DATASET ######################################################################################

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
fig_num = 0
first_image = train_images[fig_num]
print(first_image[fig_num])

# WILL SHOW IMAGE IN GRAPH OF PIXELS
#
# plt.imshow(first_image)
# fig_name = 'matplot-figure-' + (fig_num + 1)
# try:
#     plt.savefig(fig_name)
# except:
#     print('Img already on file.')

# LABELS ARE REPRESENTED AS INDEX OF AN ARRAY
# for this dataset you can make an array of strigs to represent the actual labels
labels_array = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print(train_labels[0])
