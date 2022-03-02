import tensorflow as tf

class basic_block():

    expansion = 1

    def __init__(self, out_channels, stride, downsample):

        # self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1 = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=1, strides=(stride,stride), activation=None, padding='SAME')

        # self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn1 = tf.keras.layers.BatchNormalization()

        # self.relu = nn.ReLU(inplace=True)
        self.relu = tf.keras.activations.relu
        
        # self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=3, strides=(1,1), activation=None, padding='SAME')
        
        # self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn2 = tf.keras.layers.BatchNormalization()

        # downsample is needed to bridge between different channel size
        self.downsample = downsample 


    def forward(self, sample_input):

        out = self.conv1(sample_input)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        residual = self.downsample(sample_input) if self.downsample is not None else sample_input
        # print(self.downsample is not None)
        # print("samin",sample_input)
        # print("residual",residual)
        # print("out",out)

        out += residual
        out = self.relu(out)

        return out
      
class bottleneck_block():

    expansion = 4 #channel multipler

    def __init__(self, out_channels, stride, downsample):

        # self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.conv1 = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=1, strides=(stride,stride), activation=None, padding='SAME')

        # self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn1 = tf.keras.layers.BatchNormalization()

        # self.relu = nn.ReLU(inplace=True)
        self.relu = tf.keras.activations.relu

        # self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=3, strides=(stride,stride), activation=None, padding='SAME')
        
        # self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn2 = tf.keras.layers.BatchNormalization()

        # self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * self.expansion, kernel_size=1, bias=False)
        self.conv3 = tf.keras.layers.Conv2D(filters=out_channels*bottleneck_block.expansion, kernel_size=1, strides=(1,1), activation=None, padding='SAME')
        
        # self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.bn3 = tf.keras.layers.BatchNormalization()

        # downsample is needed to bridge between different channel size
        self.downsample = downsample 


    def forward(self, sample_input):

        out = self.conv1(sample_input)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.downsample(sample_input) if self.downsample is not None else sample_input

        out += residual
        out = self.relu(out)

        return out
      
class ResNet():

    def net_block_layer(self, net_block, out_channels, num_blocks, stride=1):

        def forward(self,sample_input):

            # print(">>>>>> net_block_layer forward",self.in_channels,out_channels * net_block.expansion)

            downsample = None

            # under shortcut, if the dimension is different, change it
            if stride != 1 or self.in_channels != out_channels * net_block.expansion:
                # downsample = nn.Sequential(
                # nn.Conv2d(self.in_channels, out_channels * net_block.expansion, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(out_channels * net_block.expansion))
                downsample = tf.keras.models.Sequential()
                downsample.add(tf.keras.layers.Conv2D(filters=out_channels*net_block.expansion, kernel_size=1, strides=(stride,stride), activation=None, padding='SAME'))
                downsample.add(tf.keras.layers.BatchNormalization())

            # layers.append(net_block(self.in_channels, out_channels, stride, downsample))
            out = net_block(out_channels, stride, downsample).forward(sample_input)
        
            if net_block.expansion != 1:
                self.in_channels = out_channels * net_block.expansion
            else:
                self.in_channels = out_channels

            for i in range(1, num_blocks):
                # layers.append(net_block(self.in_channels, out_channels, 1, None))
                out = net_block(out_channels, 1, None).forward(out)
            
            return out

        return forward


    def __init__(self, net_block, layers):
        
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.pad1 = tf.keras.layers.ZeroPadding2D(padding=(3, 3))
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=(2,2), activation=None, padding='SAME')

        # self.bn1 = nn.BatchNorm2d(64)
        self.bn1 = tf.keras.layers.BatchNormalization()

        # self.relu = nn.ReLU(inplace=True)
        self.relu = tf.keras.activations.relu

        # self.maxpooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpooling = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2,2), padding='SAME')

        self.layer1 = self.net_block_layer(net_block, 64, layers[0])
        self.layer2 = self.net_block_layer(net_block, 128, layers[1], stride=2)
        self.layer3 = self.net_block_layer(net_block, 256, layers[2], stride=2)
        self.layer4 = self.net_block_layer(net_block, 512, layers[3], stride=2)

        # self.avgpooling = nn.AvgPool2d(7, stride=1)
        self.avgpooling = tf.keras.layers.AveragePooling2D(pool_size=(7, 7), strides=(1,1), data_format=None)

        self.in_channels = 64

        # # 參數初始化
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #       nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        #     elif isinstance(m, nn.BatchNorm2d):
        #       nn.init.constant_(m.weight, 1)
        #       nn.init.constant_(m.bias, 0)        



    def forward(self,num_classes,sample_input):
        x = self.pad1(sample_input)
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpooling(x)
        x = self.layer1(self,x)
        x = self.layer2(self,x)
        x = self.layer3(self,x)
        x = self.layer4(self,x)
        x = self.avgpooling(x)
        x = tf.keras.layers.Flatten()(x) # x = torch.flatten(x, start_dim=1)
        
        x = tf.keras.layers.Dense(num_classes, activation=None)(x) # self.fc = nn.Linear(512 * net_block.expansion, num_classes)

        model = tf.keras.Model(sample_input, x)
        return model

class ResNetStruct:

    construct_map = {
        18 : ResNet(basic_block, [2, 2, 2, 2]),
        34 : ResNet(basic_block, [3, 4, 6, 3]),
        50 : ResNet(bottleneck_block, [3, 4, 6, 3]),
        101 : ResNet(bottleneck_block, [3, 4, 23, 3]),
        152 : ResNet(bottleneck_block, [3, 8, 36, 3])
    }

    def __init__(self):
        pass

    def construct(self,num_layers):
        assert num_layers in ResNetStruct.construct_map
        model = ResNetStruct.construct_map[num_layers]
        return model

mymodel = ResNetStruct()


if __name__ == "__main__":
    sample_input = tf.keras.Input(shape=(224,224,3)) #input_shape = [224,224,3] #H,W,C
    out = mymodel.construct(18).forward(10,sample_input)
    print(out.summary())

