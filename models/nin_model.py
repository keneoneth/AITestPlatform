import tensorflow as tf

# source: https://d2l.ai/chapter_convolutional-modern/nin.html#nin-blocks

params = [[96, 11, "valid", 4], [256, 5, "same", 1], [384, 3, "same", 1]]  # in_channels, out_channels, kernel_size, padding, stride

class NiN():
    def __init__(self,nin_params):
        self.nin_params = nin_params

    # https://stackoverflow.com/questions/53819528/how-does-tf-keras-layers-conv2d-with-padding-same-and-strides-1-behave

    def nin_block(self,num_channels, kernel_size, padding, strides):
        layers = []
        layers.append(tf.keras.layers.Conv2D(filters=num_channels, kernel_size=kernel_size, padding=padding, strides=strides, activation='relu'))
        layers.append(tf.keras.layers.Conv2D(filters=num_channels, kernel_size=1, activation='relu'))
        layers.append(tf.keras.layers.Conv2D(filters=num_channels, kernel_size=1, activation='relu'))
        return layers
       

    def forward(self, num_classes):
        # build model
        model = tf.keras.models.Sequential()
        for layer in self.nin_block(*self.nin_params[0]):
            model.add(layer)
        model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2,2)))
        for layer in self.nin_block(*self.nin_params[1]):
            model.add(layer)
        model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2,2)))
        for layer in self.nin_block(*self.nin_params[2]):
            model.add(layer)
        model.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
        model.add(tf.keras.layers.Dropout(rate=0.5))
        for layer in self.nin_block(num_classes,self.nin_params[2][1],self.nin_params[2][2],self.nin_params[2][3]):
            model.add(layer)
        model.add(tf.keras.layers.GlobalAveragePooling2D())
        model.add(tf.keras.layers.Reshape((1, 1, 10))) #batch size is automatically inferred
        model.add(tf.keras.layers.Flatten())
        return model


mymodel = NiN(params)


if __name__ == "__main__":
    model = mymodel.forward(10)
    X = tf.random.uniform((64, 224, 224, 1))
    for layer in model.layers:
        X = layer(X)
        print(layer.__class__.__name__,'output shape:\t', X.shape)