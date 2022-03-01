import tensorflow as tf

class inceptionv1_block():
    
    
    def __init__(self, out_channels1, out_channels2_step1, out_channels2_step2, out_channels3_step1, out_channels3_step2, out_channels4):
        # super(inceptionv1_block, self).__init__()

        # self.branch1_conv = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels1, kernel_size=1),nn.ReLU(inplace=True))
        self.branch1_conv = tf.keras.layers.Conv2D(filters=out_channels1, kernel_size=1, strides=(1,1), activation=tf.keras.activations.relu, padding='SAME')

        # self.branch2_conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels2_step1, kernel_size=1),nn.ReLU(inplace=True))
        self.branch2_conv1 = tf.keras.layers.Conv2D(filters=out_channels2_step1, kernel_size=1, strides=(1,1), activation=tf.keras.activations.relu, padding='SAME')

        # self.branch2_conv2 = nn.Sequential(nn.Conv2d(in_channels=out_channels2_step1, out_channels=out_channels2_step2, kernel_size=3, padding=1),nn.ReLU(inplace=True))
        self.branch2_conv2 = tf.keras.layers.Conv2D(filters=out_channels2_step2, kernel_size=3, strides=(1,1), activation=tf.keras.activations.relu, padding='SAME')

        # self.branch3_conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels3_step1, kernel_size=1),nn.ReLU(inplace=True))
        self.branch3_conv1 = tf.keras.layers.Conv2D(filters=out_channels3_step1, kernel_size=1, strides=(1,1), activation=tf.keras.activations.relu, padding='SAME')
        
        # self.branch3_conv2 = nn.Sequential(nn.Conv2d(in_channels=out_channels3_step1, out_channels=out_channels3_step2, kernel_size=5, padding=2),nn.ReLU(inplace=True))
        self.branch3_conv2 = tf.keras.layers.Conv2D(filters=out_channels3_step2, kernel_size=5, strides=(1,1), activation=tf.keras.activations.relu, padding='SAME')

        # self.branch4_maxpooling = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_maxpooling = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1,1), padding='SAME')

        # self.branch4_conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels4, kernel_size=1),nn.ReLU(inplace=True))
        self.branch4_conv1 = tf.keras.layers.Conv2D(filters=out_channels4, kernel_size=1, strides=(1,1), activation=tf.keras.activations.relu, padding='SAME')

     
    def forward(self,layer):

        # out1 = self.branch1_conv(x)
        branch1 = self.branch1_conv(layer)

        # out2 = self.branch2_conv2(self.branch2_conv1(x))
        branch2 = self.branch2_conv1(layer)
        branch2 = self.branch2_conv2(branch2)

        # out3 = self.branch3_conv2(self.branch3_conv1(x))
        branch3 = self.branch3_conv1(layer)
        branch3 = self.branch3_conv2(branch3)

        # out4 = self.branch4_conv1(self.branch4_maxpooling(x))
        branch4 = self.branch4_maxpooling(layer)
        branch4 = self.branch4_conv1(branch4)

        # out = torch.cat([out1, out2, out3, out4], dim=1)
        # out = tf.keras.layers.concatenate([branch1, branch2, branch3, branch4],axis=3) # working?
        out = tf.keras.layers.concatenate([branch1, branch2, branch3, branch4],axis=3)
        return out
      
class auxiliary_classifiers():
    def __init__(self, out_channels):
        # super(auxiliary_classifiers, self).__init__()
        # self.avgpooling = nn.AvgPool2d(kernel_size=5, stride=3)
        self.avgpooling = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3,3))

        # self.conv = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=1)
        self.conv = tf.keras.layers.Conv2D(filters=128, kernel_size=1, strides=(1,1), activation=tf.keras.activations.relu, padding='SAME')

        # self.fc1 = nn.Linear(in_features=128*4*4, out_features=1024)
        self.fc1 = tf.keras.layers.Dense(1024, activation=tf.keras.activations.relu) 

        # self.fc2 = nn.Linear(in_features=1024, out_features=out_channels)
        self.fc2 = tf.keras.layers.Dense(out_channels, activation=None) 
     
    def forward(self):
        classifer = tf.keras.models.Sequential()
        # x = self.avgpooling(x)
        classifer.add(self.avgpooling)
        # x = F.relu(self.conv(x))
        classifer.add(self.conv)
        # x = torch.flatten(x, start_dim=1)
        classifer.add(tf.keras.layers.Flatten())
        # x = F.relu(self.fc1(x))
        classifer.add(self.fc1)
        # x = F.dropout(x, p=0.5)
        classifer.add(tf.keras.layers.Dropout(rate=0.5))
        # x = self.fc2(x)
        classifer.add(fc2)

        return classifer

class GoogLeNet():

    def __init__(self):
        # super(InceptionV1, self).__init__()
        # self.training = training
        # self.conv = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
        #             nn.ReLU(inplace=True),
        #             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1),
        #             nn.ReLU(inplace=True),
        #             nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
        #             nn.ReLU(inplace=True),
        #             nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.conv = tf.keras.models.Sequential()
        self.conv.add(tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=(2,2), activation=tf.keras.activations.relu, padding='SAME'))
        self.conv.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2,2), padding='SAME'))
        self.conv.add(tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=(1,1), activation=tf.keras.activations.relu, padding='SAME'))
        self.conv.add(tf.keras.layers.Conv2D(filters=192, kernel_size=3, strides=(1,1), activation=tf.keras.activations.relu, padding='SAME'))
        self.conv.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2,2), padding='SAME'))

        self.inception1 = inceptionv1_block(out_channels1=64, out_channels2_step1=96, out_channels2_step2=128, out_channels3_step1=16, out_channels3_step2=32, out_channels4=32)
        self.inception2 = inceptionv1_block(out_channels1=128, out_channels2_step1=128, out_channels2_step2=192, out_channels3_step1=32, out_channels3_step2=96, out_channels4=64)
        # self.maxpooling1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpooling1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2,2), padding='SAME')
        self.inception3 = inceptionv1_block(out_channels1=192, out_channels2_step1=96, out_channels2_step2=208, out_channels3_step1=16, out_channels3_step2=48, out_channels4=64)

        # if self.training == True: # disable auxiliary layers
        #     self.auxiliary1 = auxiliary_classifiers(in_channels=512,out_channels=num_classes)

        self.inception4 = inceptionv1_block(out_channels1=160, out_channels2_step1=112, out_channels2_step2=224, out_channels3_step1=24, out_channels3_step2=64, out_channels4=64)
        self.inception5 = inceptionv1_block(out_channels1=128, out_channels2_step1=128, out_channels2_step2=256, out_channels3_step1=24, out_channels3_step2=64, out_channels4=64)
        self.inception6 = inceptionv1_block(out_channels1=112, out_channels2_step1=144, out_channels2_step2=288, out_channels3_step1=32, out_channels3_step2=64, out_channels4=64)

        # if self.training == True: # disable auxiliary layers
        #     self.auxiliary2 = auxiliary_classifiers(in_channels=528,out_channels=num_classes)

        self.inception7 = inceptionv1_block(out_channels1=256, out_channels2_step1=160, out_channels2_step2=320, out_channels3_step1=32, out_channels3_step2=128, out_channels4=128)
        # self.maxpooling2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpooling2 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2,2), padding='SAME')

        self.inception8 = inceptionv1_block(out_channels1=256, out_channels2_step1=160, out_channels2_step2=320, out_channels3_step1=32, out_channels3_step2=128, out_channels4=128)
        self.inception9 = inceptionv1_block(out_channels1=384, out_channels2_step1=192, out_channels2_step2=384, out_channels3_step1=48, out_channels3_step2=128, out_channels4=128)

        # self.avgpooling = nn.AvgPool2d(kernel_size=7,stride=1)
        self.avgpooling = tf.keras.layers.AveragePooling2D(pool_size=(7, 7), strides=(1,1))

        # self.dropout = nn.Dropout(p=0.4)
        self.dropout = tf.keras.layers.Dropout(rate=0.4)

        # self.fc = nn.Linear(in_features=1024,out_features=num_classes) # move to forward function
        

    def forward(self,num_classes,sample_input):
        # build model
        layer = self.conv(sample_input)  #x = self.conv(x)
        layer = self.inception1.forward(layer) #x = self.inception1(x)
        layer = self.inception2.forward(layer) #x = self.inception2(x)
        layer = self.maxpooling1(layer) #x = self.maxpooling1(x)
        layer = self.inception3.forward(layer) #x = self.inception3(x)
        # aux1 = self.auxiliary1(x)

        layer = self.inception4.forward(layer) #x = self.inception4(x)
        layer = self.inception5.forward(layer) #x = self.inception5(x)
        layer = self.inception6.forward(layer) #x = self.inception6(x)
        # aux2 = self.auxiliary2(x)

        layer = self.inception7.forward(layer) #x = self.inception7(x)
        layer = self.maxpooling1(layer) #x = self.maxpooling2(x)
        layer = self.inception8.forward(layer) #x = self.inception8(x)
        layer = self.inception9.forward(layer) #x = self.inception9(x)

        layer = self.avgpooling(layer) #x = self.avgpooling(x)
        layer = self.dropout(layer) #x = self.dropout(x)
        layer = tf.keras.layers.Flatten()(layer) #x = torch.flatten(x, start_dim=1)
        layer = tf.keras.layers.Dense(num_classes, activation=None)(layer) #x = tf.keras.layers.

        # if self.training == True:
            # total_loss = real_loss + 0.3 * aux_loss_1 + 0.3 * aux_loss_2
            # return aux1, aux2, out
        # else:
            # return out

        model = tf.keras.Model(sample_input, layer)
        return model

mymodel = GoogLeNet()

if __name__ == "__main__":
    sample_input = tf.keras.Input(shape=(224,224,3)) #input_shape = [224,224,3] #H,W,C
    out = mymodel.forward(10,sample_input)
    print(out.summary())