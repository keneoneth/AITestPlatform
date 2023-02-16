import tensorflow as tf

class inceptionv1_block():
    
    
    def __init__(self, out_channels1, out_channels2_step1, out_channels2_step2, out_channels3_step1, out_channels3_step2, out_channels4):

        self.branch1_conv = tf.keras.layers.Conv2D(filters=out_channels1, kernel_size=1, strides=(1,1), activation=tf.keras.activations.relu, padding='SAME')

        self.branch2_conv1 = tf.keras.layers.Conv2D(filters=out_channels2_step1, kernel_size=1, strides=(1,1), activation=tf.keras.activations.relu, padding='SAME')

        self.branch2_conv2 = tf.keras.layers.Conv2D(filters=out_channels2_step2, kernel_size=3, strides=(1,1), activation=tf.keras.activations.relu, padding='SAME')

        self.branch3_conv1 = tf.keras.layers.Conv2D(filters=out_channels3_step1, kernel_size=1, strides=(1,1), activation=tf.keras.activations.relu, padding='SAME')
        
        self.branch3_conv2 = tf.keras.layers.Conv2D(filters=out_channels3_step2, kernel_size=5, strides=(1,1), activation=tf.keras.activations.relu, padding='SAME')

        self.branch4_maxpooling = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1,1), padding='SAME')

        self.branch4_conv1 = tf.keras.layers.Conv2D(filters=out_channels4, kernel_size=1, strides=(1,1), activation=tf.keras.activations.relu, padding='SAME')

     
    def forward(self,layer):

        branch1 = self.branch1_conv(layer)

        branch2 = self.branch2_conv1(layer)
        branch2 = self.branch2_conv2(branch2)

        branch3 = self.branch3_conv1(layer)
        branch3 = self.branch3_conv2(branch3)

        branch4 = self.branch4_maxpooling(layer)
        branch4 = self.branch4_conv1(branch4)

        out = tf.keras.layers.concatenate([branch1, branch2, branch3, branch4],axis=3)
        return out
      
class auxiliary_classifiers():
    def __init__(self, out_channels):
        self.avgpooling = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3,3))

        self.conv = tf.keras.layers.Conv2D(filters=128, kernel_size=1, strides=(1,1), activation=tf.keras.activations.relu, padding='SAME')

        self.fc1 = tf.keras.layers.Dense(1024, activation=tf.keras.activations.relu) 

        self.fc2 = tf.keras.layers.Dense(out_channels, activation=None) 
     
    def forward(self):
        classifer = tf.keras.models.Sequential()
        classifer.add(self.avgpooling)
        classifer.add(self.conv)
        classifer.add(tf.keras.layers.Flatten())
        classifer.add(self.fc1)
        classifer.add(tf.keras.layers.Dropout(rate=0.5))
        classifer.add(self.fc2)

        return classifer

class GoogLeNet():

    def __init__(self):
        
        self.conv = tf.keras.models.Sequential()
        self.conv.add(tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=(2,2), activation=tf.keras.activations.relu, padding='SAME'))
        self.conv.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2,2), padding='SAME'))
        self.conv.add(tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=(1,1), activation=tf.keras.activations.relu, padding='SAME'))
        self.conv.add(tf.keras.layers.Conv2D(filters=192, kernel_size=3, strides=(1,1), activation=tf.keras.activations.relu, padding='SAME'))
        self.conv.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2,2), padding='SAME'))

        self.inception1 = inceptionv1_block(out_channels1=64, out_channels2_step1=96, out_channels2_step2=128, out_channels3_step1=16, out_channels3_step2=32, out_channels4=32)
        self.inception2 = inceptionv1_block(out_channels1=128, out_channels2_step1=128, out_channels2_step2=192, out_channels3_step1=32, out_channels3_step2=96, out_channels4=64)
        self.maxpooling1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2,2), padding='SAME')
        self.inception3 = inceptionv1_block(out_channels1=192, out_channels2_step1=96, out_channels2_step2=208, out_channels3_step1=16, out_channels3_step2=48, out_channels4=64)


        self.inception4 = inceptionv1_block(out_channels1=160, out_channels2_step1=112, out_channels2_step2=224, out_channels3_step1=24, out_channels3_step2=64, out_channels4=64)
        self.inception5 = inceptionv1_block(out_channels1=128, out_channels2_step1=128, out_channels2_step2=256, out_channels3_step1=24, out_channels3_step2=64, out_channels4=64)
        self.inception6 = inceptionv1_block(out_channels1=112, out_channels2_step1=144, out_channels2_step2=288, out_channels3_step1=32, out_channels3_step2=64, out_channels4=64)


        self.inception7 = inceptionv1_block(out_channels1=256, out_channels2_step1=160, out_channels2_step2=320, out_channels3_step1=32, out_channels3_step2=128, out_channels4=128)
        self.maxpooling2 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2,2), padding='SAME')

        self.inception8 = inceptionv1_block(out_channels1=256, out_channels2_step1=160, out_channels2_step2=320, out_channels3_step1=32, out_channels3_step2=128, out_channels4=128)
        self.inception9 = inceptionv1_block(out_channels1=384, out_channels2_step1=192, out_channels2_step2=384, out_channels3_step1=48, out_channels3_step2=128, out_channels4=128)

        self.avgpooling = tf.keras.layers.AveragePooling2D(pool_size=(7, 7), strides=(1,1))

        self.dropout = tf.keras.layers.Dropout(rate=0.4)

        

    def forward(self,num_classes,sample_input):
        # build model
        layer = self.conv(sample_input)
        layer = self.inception1.forward(layer)
        layer = self.inception2.forward(layer)
        layer = self.maxpooling1(layer)
        layer = self.inception3.forward(layer)

        layer = self.inception4.forward(layer)
        layer = self.inception5.forward(layer)
        layer = self.inception6.forward(layer)

        layer = self.inception7.forward(layer)
        layer = self.maxpooling1(layer)
        layer = self.inception8.forward(layer)
        layer = self.inception9.forward(layer)

        layer = self.avgpooling(layer)
        layer = self.dropout(layer)
        layer = tf.keras.layers.Flatten()(layer)
        layer = tf.keras.layers.Dense(num_classes, activation=None)(layer)


        model = tf.keras.Model(sample_input, layer)
        return model

mymodel = GoogLeNet()

if __name__ == "__main__":
    sample_input = tf.keras.Input(shape=(224,224,3)) #input_shape = [224,224,3] #H,W,C
    out = mymodel.forward(10,sample_input)
    print(out.summary())