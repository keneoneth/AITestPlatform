# Dense block has a structure of batch norm, relu, and 3x3 conv2d
import tensorflow as tf

class basic_layer():
    def __init__(self, out_channels):
        # super(basic_layer, self).__init__()      
        # self.basic = nn.Sequential(nn.BatchNorm2d(in_channels), 
        # nn.ReLU(inplace=True),
        # nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False))

        self.basic = tf.keras.models.Sequential()
        self.basic.add(tf.keras.layers.BatchNormalization())
        self.basic.add(tf.keras.layers.ReLU())
        self.basic.add(tf.keras.layers.Conv2D(filters=out_channels, kernel_size=3, strides=(1,1), activation=None, padding='SAME'))

    def forward(self, sample_input):
        out = self.basic(sample_input)
        # out = torch.cat([out, x], dim=1) 
        out = tf.keras.layers.concatenate([out, sample_input],axis=3)
        return out
      
# 為了降低通道維度、減少參數計算量，採用 Bottleneck (Batch Normalization、ReLU、1x1 卷積層、Batch Normalization、ReLU、3x3 卷積層)
class bottleneck_layer():
    def __init__(self, bottleneck_size, growth_rate, drop_rate):
        # super(bottleneck_layer, self).__init__()      
        # self.bottleneck = nn.Sequential(nn.BatchNorm2d(in_channels),
        #               nn.ReLU(inplace=True),
        #               nn.Conv2d(in_channels=in_channels, out_channels=bottleneck_size*growth_rate, kernel_size=1, padding=0, bias=False),
        #               nn.BatchNorm2d(bottleneck_size*growth_rate),
        #               nn.ReLU(inplace=True),
        #               nn.Conv2d(in_channels=bottleneck_size*growth_rate, out_channels=growth_rate, kernel_size=3, padding=1, bias=False))
        self.bottleneck = tf.keras.models.Sequential()
        self.bottleneck.add(tf.keras.layers.BatchNormalization())
        self.bottleneck.add(tf.keras.layers.ReLU())
        self.bottleneck.add(tf.keras.layers.Conv2D(filters=bottleneck_size*growth_rate, kernel_size=3, strides=(1,1), activation=None, padding='SAME'))
        self.bottleneck.add(tf.keras.layers.BatchNormalization())
        self.bottleneck.add(tf.keras.layers.ReLU())
        self.bottleneck.add(tf.keras.layers.Conv2D(filters=growth_rate, kernel_size=3, strides=(1,1), activation=None, padding='SAME'))
        

        self.drop_rate = drop_rate
        # self.dropout = nn.Dropout(p=self.drop_rate)
        self.dropout = tf.keras.layers.Dropout(rate=self.drop_rate)

    def forward(self, sample_input):
        out = self.bottleneck(sample_input)
        if self.drop_rate > 0:
            out = self.dropout(out)

        # out = torch.cat([out, x], dim=1) 
        out = tf.keras.layers.concatenate([out, sample_input],axis=3)
        return out
   
class DenseNet():

    def dense_block(self, bottleneck_size, growth_rate, drop_rate, num_layers):
        block = []
        for _ in range(num_layers):
            # block.append(bottleneck_layer(in_channels + i*growth_rate, bottleneck_size, growth_rate, drop_rate))
            block.append(bottleneck_layer(bottleneck_size, growth_rate, drop_rate))

        # return nn.Sequential(*block)
        return block
    def __init__(self, out_channels, growth_rate, num_layers):
        # super(DenseNet, self).__init__()      
        bottleneck_size = 4
        drop_rate = 0.0

        # self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=7, stride=2, padding=3, bias=False),
        #             nn.BatchNorm2d(out_channels),
        #             nn.ReLU(inplace=True))
        self.conv1 = tf.keras.models.Sequential()
        self.conv1.add(tf.keras.layers.ZeroPadding2D(padding=(3, 3)))
        self.conv1.add(tf.keras.layers.Conv2D(filters=out_channels, kernel_size=7, strides=(2,2), activation=None, padding='SAME'))
        self.conv1.add(tf.keras.layers.BatchNormalization())
        self.conv1.add(tf.keras.layers.ReLU())


        # self.maxpooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpooling = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2,2), padding='SAME')

        layers = []
        block_in_channels = out_channels
        for i, num_layer in enumerate(num_layers):
            layers.append(self.dense_block(bottleneck_size, growth_rate, drop_rate, num_layer))
        block_in_channels += num_layer*growth_rate

        if i != len(num_layers)-1:
          layers.append(self.transition_layer(block_in_channels, block_in_channels // 2))
          block_in_channels = block_in_channels // 2

        self.blocks = layers
        
        # self.bn = nn.BatchNorm2d(block_in_channels)
        self.bn = tf.keras.layers.BatchNormalization()
        # self.relu = nn.ReLU(inplace=True)
        self.relu = tf.keras.layers.ReLU()
        # self.GAP_pooling = nn.AvgPool2d(7, stride=1)
        self.GAP_pooling = tf.keras.layers.AveragePooling2D(pool_size=(7,7), strides=(1,1), data_format=None)

        # self.fc = nn.Linear(block_in_channels, num_classes)

    

    def transition_layer(self, out_channels):
        # transition = nn.Sequential(nn.BatchNorm2d(in_channels),
        #               nn.ReLU(inplace=True),
        #               nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, bias=False),
        #               nn.AvgPool2d(kernel_size=2,stride=2))
        transition = tf.keras.models.Sequential()
        transition.add(tf.keras.layers.BatchNormalization())
        transition.add(tf.keras.layers.ReLU())
        transition.add(tf.keras.layers.Conv2D(filters=out_channels, kernel_size=1, strides=(1,1), activation=None, padding='SAME'))
        transition.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2), data_format=None))
        return transition

    def forward(self, num_classes, sample_input):
        x = self.conv1(sample_input)
        x = self.maxpooling(x)
        for block in self.blocks:
            if isinstance(block,list):
                for elem in block:
                    x = elem.forward(x)
            else:
                x = block(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.GAP_pooling(x)
        x = tf.keras.layers.Flatten()(x) #x = torch.flatten(x, 1)
        x = tf.keras.layers.Dense(num_classes, activation=None)(x) #x = self.fc(x)

        model = tf.keras.Model(sample_input, x)
        return model
      
class DenseNetStruct():

    construct_map = {
        101 : DenseNet(64, 32, [6, 12, 64, 48]),
        121 : DenseNet(64, 32, [6, 12, 24, 16]),
        169 : DenseNet(64, 32, [6, 12, 32, 32]),
        201 : DenseNet(64, 32, [6, 12, 48, 32])
    }

    def __init__(self):
        pass

    def construct(self,num_layers):
        assert num_layers in DenseNetStruct.construct_map
        model = DenseNetStruct.construct_map[num_layers]
        return model

mymodel = DenseNetStruct()

if __name__ == "__main__":
    sample_input = tf.keras.Input(shape=(224,224,3)) #input_shape = [224,224,3] #H,W,C
    out = mymodel.construct(101).forward(10,sample_input)
    print(out.summary())
