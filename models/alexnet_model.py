# https://www.analyticsvidhya.com/blog/2021/03/introduction-to-the-architecture-of-alexnet/

import tensorflow as tf

class AlexNet():
    def __init__(self):
        self.pad1_input = tf.keras.layers.ZeroPadding2D(padding=(1, 1))
        self.pad2_input = tf.keras.layers.ZeroPadding2D(padding=(2, 2))
        
        ###CONV1
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, padding=2, stride=4)
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=11, strides=(4,4), activation=tf.keras.activations.relu)
        
        ###CONV2
        # self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2)
        self.conv2 = tf.keras.layers.Conv2D(filters=192, kernel_size=5, strides=(1,1), activation=tf.keras.activations.relu)
        
        ###CONV3
        # self.conv3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1)
        self.conv3 = tf.keras.layers.Conv2D(filters=384, kernel_size=3, strides=(1,1), activation=tf.keras.activations.relu)
        
        ###CONV4        
        # self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.conv4 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1,1), activation=tf.keras.activations.relu)
        
        ###CONV5
        # self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv5 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1,1), activation=tf.keras.activations.relu)

        # self.fc1 = nn.Linear(in_features=256*6*6, out_features=4096)
        self.fc1 = tf.keras.layers.Dense(units=4096, activation=tf.keras.activations.relu) 
        # self.fc2 = nn.Linear(in_features=4096, out_features=1024)
        self.fc2 = tf.keras.layers.Dense(units=1000, activation=tf.keras.activations.relu) 
       

    def forward(self, num_classes):
        # build model
        model = tf.keras.models.Sequential()
        model.add(self.pad2_input)
        model.add(self.conv1) #x = F.relu(self.conv1(x))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2,2))) #x = F.max_pool2d(x, kernel_size=3, stride=2)

        model.add(self.pad2_input)
        model.add(self.conv2) #x = F.relu(self.conv2(x))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2,2))) #x = F.max_pool2d(x, kernel_size=3, stride=2)

        model.add(self.pad1_input)
        model.add(self.conv3) #x = F.relu(self.conv3(x))

        model.add(self.pad1_input)
        model.add(self.conv4) #x = F.relu(self.conv4(x))

        model.add(self.pad1_input)
        model.add(self.conv5) #x = F.relu(self.conv5(x))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2,2))) #x = F.max_pool2d(x, kernel_size=3, stride=2)

       
        model.add(tf.keras.layers.Flatten()) # x = torch.flatten(x, start_dim=1)
        model.add(self.fc1) #x = F.relu(self.fc1(x))
        model.add(tf.keras.layers.Dropout(rate=0.5)) #x = F.dropout(x, p=0.5) 
        model.add(self.fc2) #x = F.relu(self.fc2(x))
        model.add(tf.keras.layers.Dropout(rate=0.5)) #x = F.dropout(x, p=0.5)

        # self.fc3 = nn.Linear(in_features=1024, out_features=num_classes)
        self.fc3 = tf.keras.layers.Dense(num_classes, activation=None) 
        model.add(self.fc3)

        #noise shape https://stackoverflow.com/questions/46585069/keras-dropout-with-noise-shape like broadcasting

        return model

mymodel = AlexNet()