import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPool2D, Conv2DTranspose
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomContrast


class ConvsLayer(tf.keras.layers.Layer):
    """Double 2D convolution - BatchNorm - ReLU activation.

    Layer initilizated with the same num of output filters, kernel size, and initilization type in both convolutions.

    input: a tensor of shape (batch_size, height, width, channels).
    output: a tensor of shape (batch_size, height, width, channels_out).
    """

    def __init__(self, channels_out, kernel_size, name="Convs_Layer"):
        super(ConvsLayer, self).__init__(name=name)

        ########## YOUR CODE ############

        self.conv1 = Conv2D(
            filters=channels_out, kernel_size=kernel_size, padding="same", kernel_initializer="he_normal"
        )
        self.norm1 = BatchNormalization()
        self.conv2 = Conv2D(
            filters=channels_out, kernel_size=kernel_size, padding="same", kernel_initializer="he_normal"
        )
        self.norm2 = BatchNormalization()
        self.actv = ReLU()

        ##########    END    ############

    def call(self, inp):

        ########## YOUR CODE ############

        x = self.actv(self.norm1(self.conv1(inp)))
        out = self.actv(self.norm2(self.conv2(x)))

        return out

        ##########    END    ############


class EncoderLayer(tf.keras.layers.Layer):
    """Encoder Layer (down-sampling):

    input:
    - a tensor of shape (batch_size, height, width, channels)

    outputs:
    - a tensor of shape (batch_size, height//2, width//2, channels)
    - a tensor to concatenate to the respective Decoder Layer of shape (batch_size, height, width, channels)
      obtained from a double convolution layer.
    """

    def __init__(self, channels_in, kernel_size, name="EncoderLayer"):
        super(EncoderLayer, self).__init__(name=name)

        ########## YOUR CODE ############

        self.double_conv = ConvsLayer(channels_in, kernel_size)
        self.down = MaxPool2D(pool_size=(2, 2))

        ##########    END    ############

    def call(self, inp):

        ########## YOUR CODE ############

        out1 = self.double_conv(inp)
        out2 = self.down(out1)

        return out1, out2

        ##########    END    ############


class DecoderLayer(tf.keras.layers.Layer):
    """Decoder Layer (up-sampling):

    inputs:
    - a tensor of shape (batch_size, height, width, channels)
    - a tensor to concatenate from the respective Encoder Layer.

    output:
    - a tensor of shape (batch_size, height*2, width*2, channels)
    """

    def __init__(self, channels_in, kernel_size, name="DecoderLayer"):
        super(DecoderLayer, self).__init__(name=name)

        self.channels_in = channels_in
        ########## YOUR CODE ############

        self.up = Conv2DTranspose(
            filters=channels_in, kernel_size=kernel_size, strides=2, padding="same", kernel_initializer="he_normal"
        )
        self.norm = BatchNormalization()
        self.actv = ReLU()
        self.double_conv = ConvsLayer(channels_in, kernel_size)

        ##########    END    ############

    def call(self, inp, lay):

        ########## YOUR CODE ############

        x = self.actv(self.norm(self.up(inp)))
        x = tf.concat([lay, x], axis=-1)

        out = self.double_conv(x)

        # This is a check for the size of the concatenation. Make sure to call the output of the concatenation 'x'
        assert x.shape[-1] == self.channels_in * 2

        return out

        ##########    END    ############


class AugmentLayer(tf.keras.layers.Layer):
    """The Augmentation Layer.

    inputs:
    - flip_mode: a string indicating which flip mode to use.
                 Can be "horizontal", "vertical", or "horizontal_and_vertical".

    - rotate_factor: a float represented as fraction of 2*pi, or a tuple of size 2 representing
                     lower and upper bound for rotating clockwise and counter-clockwise.

    - zoom_factor: a float represented as fraction of value, or a tuple of size 2 representing
                   lower and upper bound for zooming vertically.

    - contrast_factor: a positive float represented as fraction of value,
                       or a tuple of size 2 representing lower and upper bound.

    outputs:
    - Augmented image and segmentation using randomly the flipping, rotation, and/or the scaling technique.
    """

    def __init__(self, flip_mode, rotate_factor, zoom_factor, contrast_factor, seed=1234, name="Augmentation_Layer"):
        super(AugmentLayer, self).__init__(name=name)

        SEEDS = tf.random.uniform(shape=(4,), maxval=123456, dtype=tf.int32)

        self.flip_X = RandomFlip(mode=flip_mode, seed=SEEDS[0])
        self.flip_y = RandomFlip(mode=flip_mode, seed=SEEDS[0])

        self.rotate_X = RandomRotation(
            rotate_factor, fill_mode="constant", interpolation="bilinear", seed=SEEDS[1], fill_value=0.0
        )
        self.rotate_y = RandomRotation(
            rotate_factor, fill_mode="constant", interpolation="nearest", seed=SEEDS[1], fill_value=0.0
        )

        self.zoom_X = RandomZoom(
            zoom_factor, fill_mode="constant", interpolation="bilinear", seed=SEEDS[2], fill_value=0.0
        )
        self.zoom_y = RandomZoom(
            zoom_factor, fill_mode="constant", interpolation="nearest", seed=SEEDS[2], fill_value=0.0
        )

        self.contrast = RandomContrast(contrast_factor, seed=SEEDS[3])

    def call(self, X, y):

        ########## YOUR CODE ############
        # flipping
        X, y = self.flip_X(X), self.flip_y(y)

        # rotation
        X, y = self.rotate_X(X), self.rotate_y(y)

        # zoom
        X, y = self.zoom_X(X), self.zoom_y(y)

        # contrast
        X = self.contrast(X)

        ##########    END    ############

        return X, y
