import tensorflow as tf

class basic_layer():
    def __init__(self, out_channels):
        self.basic = tf.keras.models.Sequential()
        self.basic.add(tf.keras.layers.BatchNormalization())
        self.basic.add(tf.keras.layers.ReLU())
        self.basic.add(tf.keras.layers.Conv2D(filters=out_channels, kernel_size=3, strides=(1,1), activation=None, padding='SAME'))

    def forward(self, sample_input):
        out = self.basic(sample_input)
        out = tf.keras.layers.concatenate([out, sample_input],axis=3)
        return out
      
class bottleneck_layer():
    def __init__(self, bottleneck_size, growth_rate, drop_rate):
        self.bottleneck = tf.keras.models.Sequential()
        self.bottleneck.add(tf.keras.layers.BatchNormalization())
        self.bottleneck.add(tf.keras.layers.ReLU())
        self.bottleneck.add(tf.keras.layers.Conv2D(filters=bottleneck_size*growth_rate, kernel_size=3, strides=(1,1), activation=None, padding='SAME'))
        self.bottleneck.add(tf.keras.layers.BatchNormalization())
        self.bottleneck.add(tf.keras.layers.ReLU())
        self.bottleneck.add(tf.keras.layers.Conv2D(filters=growth_rate, kernel_size=3, strides=(1,1), activation=None, padding='SAME'))
        

        self.drop_rate = drop_rate
        self.dropout = tf.keras.layers.Dropout(rate=self.drop_rate)

    def forward(self, sample_input):
        out = self.bottleneck(sample_input)
        if self.drop_rate > 0:
            out = self.dropout(out)

        out = tf.keras.layers.concatenate([out, sample_input],axis=3)
        return out
   
class DenseNet():

    def dense_block(self, bottleneck_size, growth_rate, drop_rate, num_layers):
        block = []
        for _ in range(num_layers):
            block.append(bottleneck_layer(bottleneck_size, growth_rate, drop_rate))

        return block
    def __init__(self, out_channels, growth_rate, num_layers):
        # super(DenseNet, self).__init__()      
        bottleneck_size = 4
        drop_rate = 0.0

        self.conv1 = tf.keras.models.Sequential()
        self.conv1.add(tf.keras.layers.ZeroPadding2D(padding=(3, 3)))
        self.conv1.add(tf.keras.layers.Conv2D(filters=out_channels, kernel_size=7, strides=(2,2), activation=None, padding='SAME'))
        self.conv1.add(tf.keras.layers.BatchNormalization())
        self.conv1.add(tf.keras.layers.ReLU())


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
        
        self.bn = tf.keras.layers.BatchNormalization()

        self.relu = tf.keras.layers.ReLU()

        self.GAP_pooling = tf.keras.layers.AveragePooling2D(pool_size=(7,7), strides=(1,1), data_format=None)


    

    def transition_layer(self, out_channels):
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
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(num_classes, activation=None)(x)

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
