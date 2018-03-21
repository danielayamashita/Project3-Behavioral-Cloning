#==============================================================================
# Selfing-drive car 
# Author: Daniela Yassuda Yamashita
#February 26th 2018
#==============================================================================

from IPython import get_ipython
get_ipython().magic('reset -sf')
import csv
import cv2  
import numpy as np
import os.path

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda ,Cropping2D,Dropout
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.models import load_model


# Unzip the file containing all the images
from zipfile import ZipFile
# specifying the zip file name
file_name = "../Teste6.zip"
 
# opening the zip file in READ mode
with ZipFile(file_name, 'r') as zap:
    # printing all the contents of the zip file
    zap.printdir()
 
    # extracting all the files
    print('Extracting all the files now...')
    zap.extractall()
    print('Done!')
	
#==============================================================================
#----------------- IMPORT THE DATA --------------------------------------------
#==============================================================================
lines = []
with open('Teste6/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
    
numLines = 0
images = []
measurements = []
for line in lines:
    numLines= numLines+1
    #center images
    source_path = line[0]
    filename = source_path.split('\\')[-1]
    current_path = 'Teste4/IMG/' +filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

    if (numLines >10000):
        break

#==============================================================================
#----------------- DATA AUGMENTATION ------------------------------------------
#==============================================================================
augmented_measurements = []
for image, measurement  in zip(images,measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)
	

#Split the data into training and validating dat
total = len(augmented_measurements)
n_train= np.floor(total*0.8).astype(int)

X_train, y_train = np.array(augmented_images[0:n_train]), np.array( augmented_measurements[0:n_train])
X_test, y_test = np.array(augmented_images[n_train+1:-1]), np.array( augmented_measurements[n_train+1:-1])

#==============================================================================
#----------------- MODEL DEFINITION -------------------------------------------
#==============================================================================	
model = Sequential()
model.add(Lambda(lambda x:x/255-0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping =((70,25),(0,0))))
model.add(Convolution2D(12,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,2,2,activation='relu'))
model.add(Convolution2D(64,2,2,activation='relu'))
model.add(Convolution2D(64,2,2,activation='relu'))

model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss ='mse', optimizer = 'adam')


#Split the data into training and validating data
total = len(augmented_measurements)
n_train= np.floor(total*0.9).astype(int)
X_train, y_train = np.array(augmented_images[0:n_train]), np.array(augmented_measurements[0:n_train])
X_test, y_test = np.array(augmented_images[n_train+1:-1]), np.array(augmented_measurements[n_train+1:-1])


model.fit(X_train, y_train, shuffle=True, nb_epoch = 5)
print("Finished")
model.save('model.h5')
#Evaluate the model

print("Testing")
metrics = model.evaluate( x=X_test, y=y_test)

for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    if metrics.size ==1:
        metric_value  = metrics
    else:
        metric_value = metrics[metric_i]    
    print('{}: {}'.format(metric_name, metric_value))
    
model.save('model2.h5')

