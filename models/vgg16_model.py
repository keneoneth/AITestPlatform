from statistics import mode
import tensorflow as tf

# https://www.geeksforgeeks.org/vgg-16-cnn-model/

vgg_arch16 = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]

class VGGNet():

    def vgg_block(self, vgg_arch):
        layers = []
        in_channels = 3 

        for v in vgg_arch:
            if v == "M":
                layers.append(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2)))
            else:
                layers.append(tf.keras.layers.ZeroPadding2D(padding=(1, 1)))
                layers.append(tf.keras.layers.Conv2D(filters=in_channels, kernel_size=3, strides=(1,1), activation=tf.keras.activations.relu)) 
                in_channels = v

        return layers  

    def __init__(self, vgg_arch):
        self.features = self.vgg_block(vgg_arch)

        self.fc1 = tf.keras.layers.Dense(units=4096, activation=tf.keras.activations.relu) 

        self.fc2 = tf.keras.layers.Dense(units=1000, activation=tf.keras.activations.relu) 
             

    def forward(self, num_classes):

        # build model
        model = tf.keras.models.Sequential()

        for layer in self.features:
            model.add(layer)

        model.add(tf.keras.layers.Flatten())
        model.add(self.fc1)
        model.add(self.fc2)
        self.fc3 = tf.keras.layers.Dense(num_classes, activation=None)
        model.add(self.fc3)

        return model

mymodel = VGGNet(vgg_arch16)