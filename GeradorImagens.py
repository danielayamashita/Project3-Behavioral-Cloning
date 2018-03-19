#==============================================================================
# Selfing-drive car 
# Author: Daniela Yassuda Yamashita
#February 26th 2018
#==============================================================================

import csv
import cv2  
import numpy as np
import os.path
import matplotlib.pyplot as plt


lines = []
with open('../Training-Data/Teste4/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

numLines = 0
images = []
measurements = []

for line in lines:
    numLines= numLines+1
    source_path = line[0]
    filename = source_path.split('\\')[-1]
    current_path = '../Training-Data/Teste4/IMG/' +filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    if numLines > 0:
        break

plt.figure()
plt.imshow(image)


