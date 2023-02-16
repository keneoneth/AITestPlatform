# https://www.analyticsvidhya.com/blog/2021/03/introduction-to-the-architecture-of-alexnet/

import tensorflow as tf

class AlexNet():
    def __init__(self):
        self.pad1_input = tf.keras.layers.ZeroPadding2D(padding=(1, 1))
        self.pad2_input = tf.keras.layers.ZeroPadding2D(padding=(2, 2))
        
        ###CONV1
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=11, strides=(4,4), activation=tf.keras.activations.relu)
        
        ###CONV2
        self.conv2 = tf.keras.layers.Conv2D(filters=192, kernel_size=5, strides=(1,1), activation=tf.keras.activations.relu)
        
        ###CONV3
        self.conv3 = tf.keras.layers.Conv2D(filters=384, kernel_size=3, strides=(1,1), activation=tf.keras.activations.relu)
        
        ###CONV4        
        self.conv4 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1,1), activation=tf.keras.activations.relu)
        
        ###CONV5
        self.conv5 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1,1), activation=tf.keras.activations.relu)

        self.fc1 = tf.keras.layers.Dense(units=4096, activation=tf.keras.activations.relu) 

        self.fc2 = tf.keras.layers.Dense(units=1000, activation=tf.keras.activations.relu) 
       

    def forward(self, num_classes):
        # build model
        model = tf.keras.models.Sequential()

        model.add(self.pad2_input)
        model.add(self.conv1)
        model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2,2)))

        model.add(self.pad2_input)
        model.add(self.conv2)
        model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2,2)))

        model.add(self.pad1_input)
        model.add(self.conv3)

        model.add(self.pad1_input)
        model.add(self.conv4)

        model.add(self.pad1_input)
        model.add(self.conv5)
        model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2,2)))

       
        model.add(tf.keras.layers.Flatten())
        model.add(self.fc1)
        model.add(tf.keras.layers.Dropout(rate=0.5)) 
        model.add(self.fc2)
        model.add(tf.keras.layers.Dropout(rate=0.5))

        self.fc3 = tf.keras.layers.Dense(num_classes, activation=None) 
        model.add(self.fc3)

        return model

mymodel = AlexNet()