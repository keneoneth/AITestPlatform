import tensorflow as tf
import numpy as np

''' summary output
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

        ###CONV1
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2, stride=1) #keras version
        self.pad_input = tf.keras.layers.ZeroPadding2D(padding=(2, 2))
        self.conv1 = tf.keras.layers.Conv2D(filters=6, kernel_size=5, strides=(1,1), activation=tf.keras.activations.sigmoid)

        ###CONV2
        # self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation=tf.keras.activations.sigmoid)

        # self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc1 = tf.keras.layers.Dense(120, activation=tf.keras.activations.sigmoid) 
        # self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc2 = tf.keras.layers.Dense(84, activation=tf.keras.activations.sigmoid) 
        # self.fc3 = nn.Linear(in_features=84, out_features=10)
        self.fc3 = tf.keras.layers.Dense(10, activation=None) 

    def forward(self):
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
        model.add(self.fc3)

        # set loss function
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # compile model
        model.compile(optimizer='adam',loss=loss_fn,metrics=['accuracy'])

        return model

mymodel = LeNet().forward()