import os
import matplotlib.pyplot as plt
import cv2

# Set path to wherever your processed dataset is
path = ""
train_data = []
train_labels = []
for i in os.listdir(path)[1:]:
    p = path + i
    print("Reading "+i)
    for j in sorted(os.listdir(p)):
        print(p+'/'+j)
        img = cv2.imread(p+'/'+j)
        img = cv2.resize(img, (490,490))
        train_data.append(img)
        train_labels.append(i)
        
    print(i+" done")

# Create data and labels np arrays
import numpy as np

data = np.array(train_data)
labels = np.array(train_labels)

# Label Encode Output Alphabets
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(labels)

labels = le.transform(labels)
classes = le.classes_

# Split into Train and Test
from sklearn.model_selection import train_test_split

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size = 0.3, random_state = 0)

# Build a CNN
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(490, 490, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(24))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Fit Model to dataset
model.fit(train_data, train_labels, epochs = 10, validation_data = (test_data, test_labels))

test_loss, test_acc = model.evaluate(test_data, test_labels, verbose = 2)
print("Test Loss = {}\nTest Accuracy = {}".format(test_loss, test_accuracy))

model.save('/content/drive/My Drive/ISL.h5')
