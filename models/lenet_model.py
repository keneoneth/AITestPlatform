import tensorflow as tf
import numpy as np

''' summary output: assume input shape is 28,28,1 as in mnist
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
zero_padding2d (ZeroPadding2 (None, 32, 32, 1)         0
_________________________________________________________________
conv2d (Conv2D)              (None, 28, 28, 6)         156
_________________________________________________________________
average_pooling2d (AveragePo (None, 14, 14, 6)         0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 10, 10, 16)        2416
_________________________________________________________________
average_pooling2d_1 (Average (None, 5, 5, 16)          0
_________________________________________________________________
flatten (Flatten)            (None, 400)               0
_________________________________________________________________
dense (Dense)                (None, 120)               48120
_________________________________________________________________
dense_1 (Dense)              (None, 84)                10164
_________________________________________________________________
dense_2 (Dense)              (None, 10)                850
=================================================================
Total params: 61,706
Trainable params: 61,706
Non-trainable params: 0
_________________________________________________________________
'''


class LeNet():
    def __init__(self):
        self.pad_input = tf.keras.layers.ZeroPadding2D(padding=(2, 2))

        ###CONV1
        self.conv1 = tf.keras.layers.Conv2D(filters=6, kernel_size=5, strides=(1,1), activation=tf.keras.activations.sigmoid)

        ###CONV2
        self.conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation=tf.keras.activations.sigmoid)

        self.fc1 = tf.keras.layers.Dense(120, activation=tf.keras.activations.sigmoid) 
        self.fc2 = tf.keras.layers.Dense(84, activation=tf.keras.activations.sigmoid) 
        

    def forward(self,num_classes):
        # build model
        model = tf.keras.models.Sequential()
        model.add(self.pad_input)
        model.add(self.conv1)
        model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2)))
        model.add(self.conv2)
        model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2)))
        model.add(tf.keras.layers.Flatten())
        model.add(self.fc1)
        model.add(self.fc2)
        
        ### output layer
        self.fc3 = tf.keras.layers.Dense(num_classes, activation=None) 
        model.add(self.fc3)

        return model

mymodel = LeNet()