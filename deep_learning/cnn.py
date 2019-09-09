# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 00:05:18 2019

@author: User
"""

################################### Importing libraries###################################
# preprocessing train the model
from keras.preprocessing.image import ImageDataGenerator
# build model
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
# preprocessing to make new prediction
import numpy as np
from keras.preprocessing import image


################################### Pre-processing ###################################
######Data Augmentation: expand the size of a training dataset by creating modified versions of images in the dataset
### training_set
# parameters to generate new images
train_datagen = ImageDataGenerator(rescale = 1./255,		#fit the same scale: set all values between 0 et 1
                                   shear_range = 0.2,		#create new images by transvection (change the angle of view)
                                   zoom_range = 0.2,		#create new images by zoom 		
                                   horizontal_flip = True)	#create new images by horizontal flip
#generate new training_set images
training_set = train_datagen.flow_from_directory(r"C:\Users\User\desk\machine_learning\deeplearning-master\deeplearning-master\Part 2 - Convolutional_Neural_Networks\dataset\training_set",
                                                 target_size = (64, 64),	#size of the new images
                                                 batch_size = 32,			#update the newtwork after a bacth of 32 oberservations
                                                 class_mode = 'binary')		#binary variable

###test_datagen
# parameters to generate new images
test_datagen = ImageDataGenerator(rescale = 1./255) 		#fit the same scale: set all values between 0 et 1
#generate new test_data images
test_set = test_datagen.flow_from_directory(r"C:\Users\User\desk\machine_learning\deeplearning-master\deeplearning-master\Part 2 - Convolutional_Neural_Networks\dataset\test_set",
                                            target_size = (64, 64),	#size of the new images
                                            batch_size = 32,		#update the newtwork after a bacth of 32 oberservations	
                                            class_mode = 'binary')	#binary variable (only tow category)
											
################################### Build the  CNN###################################
# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

################################### Run the  model: Fitting the CNN to the images ###################################

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)

################################### Making new predictions ###################################
# preprocessing the new test prediction
test_image = image.load_img(r"C:\Users\User\desk\machine_learning\deeplearning-master\deeplearning-master\Part 2 - Convolutional_Neural_Networks\dataset\single_prediction\cat_or_dog_1.jpg", target_size = (64, 64))
# preprocessing the new test prediction
test_image = image.img_to_array(test_image)			# transform the original image into a new image in 3 dimensions.
test_image = np.expand_dims(test_image, axis = 0)	# add dimension that tells us in which group the image is (axis = 0 mean one group)

result = classifier.predict(test_image)				# return 0 or 1 

training_set.class_indices							# indicate if 0 is cat or dog
# based on class_indices say if it's a dog (1) or a cat (0)
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
