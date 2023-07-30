import tensorflow as tf
import keras
from keras import layers


def CasnetGenerator(
    input_shape,
    out_channels,
    out_activation="sigmoid",
    n_blocks=2,
    use_instance_norm=True,
    name=None,
    **kwargs,
) -> keras.Model:
    """Create a CasNet Generator using UBlocks"""
    input_tensor = layers.Input(shape=input_shape)
    x = input_tensor
    for i in range(n_blocks):
        x = u_block(x, block_prefix=f"UBlock{i+1}", use_instance_norm=use_instance_norm)

    # Last operation: 1x1 conv to map the image to the input n_channels.
    x = layers.Conv2D(
        out_channels,  # input_tensor.shape[-1]
        kernel_size=(1, 1),
        name="logits",
    )(x)
    output = layers.Activation(out_activation, name="preds")(x)

    return keras.Model(inputs=input_tensor, outputs=output, name=name)


def Generator(
    input_shape,
    out_channels,
    out_activation="sigmoid",
    use_resize_convolution=False,
    use_dropout=False,
    name=None,
    **kwargs,
) -> keras.Model:
    # Layer 1: Input
    input_img = layers.Input(shape=input_shape)
    x = ReflectionPadding2D((3, 3))(input_img)
    x = c7Ak(x, 32)

    # Layer 2-3: Downsampling
    x = dk(x, 64)
    x = dk(x, 128)

    # Layers 4-12: Residual blocks
    for _ in range(4, 13):
        x = Rk(x, use_dropout=use_dropout)

    # Layer 13:14: Upsampling
    x = uk(x, 64, use_resize_convolution=use_resize_convolution)
    x = uk(x, 32, use_resize_convolution=use_resize_convolution)

    # Layer 15: Output
    x = ReflectionPadding2D((3, 3))(x)
    x = layers.Conv2D(
        out_channels, kernel_size=7, strides=1, padding="valid", use_bias=True
    )(x)
    x = layers.Activation(out_activation)(x)  # !: Replaced the fixed tanh activation.

    return keras.Model(inputs=input_img, outputs=x, name=name)


def Discriminator(
    input_shape,
    use_bias=True,
    use_patchgan=True,
    name=None,
    **kwargs,
) -> keras.Model:
    # Input
    input_img = layers.Input(shape=input_shape)

    # Layers 1-4
    x = ck(
        input_img, 64, use_normalization=False, use_bias=use_bias
    )  #  Instance normalization is not used for this layer)
    x = ck(x, 128, use_normalization=True, use_bias=use_bias)
    x = ck(x, 256, use_normalization=True, use_bias=use_bias)
    x = ck(x, 512, use_normalization=True, use_bias=use_bias)

    # Layer 5: Output
    if use_patchgan:
        x = layers.Conv2D(
            filters=1, kernel_size=4, strides=1, padding="same", use_bias=True
        )(x)
    else:
        x = layers.Flatten()(x)
        x = layers.Dense(1)(x)

    return keras.Model(inputs=input_img, outputs=x, name=name)


# Main blocks.
# ------------------------------------------------------------------------------


def u_block(
    input_tensor,
    encoder_num_filters=[64, 128, 256, 512, 512, 512, 512, 512],
    decoder_num_filters=[512, 1024, 1024, 1024, 1024, 512, 256, 128],
    use_instance_norm=True,
    block_prefix="",
):
    x = input_tensor
    encoder_blocks_outputs = []

    # Encoder path.
    for i, num_filters in enumerate(encoder_num_filters):
        x = layers.Conv2D(
            num_filters,
            kernel_size=4,
            strides=2,
            padding="same",
            name=f"{block_prefix}_EncoderBlock{i+1}-Conv",
        )(x)
        if use_instance_norm:
            x = layers.GroupNormalization(
                groups=num_filters,
                name=f"{block_prefix}_EncoderBlock{i+1}-Instancenorm",
            )(x)
        else:
            x = layers.BatchNormalization(
                name=f"{block_prefix}_EncoderBlock{i+1}-Batchnorm"
            )(x)
        x = layers.Activation("relu", name=f"{block_prefix}_EncoderBlock{i+1}-ReLU")(x)

        # Append the encoder blocks outputs, except for the last one.
        if i != len(encoder_num_filters) - 1:
            encoder_blocks_outputs.append(x)

    # Decoder path.
    for i, num_filters in enumerate(decoder_num_filters):
        x = layers.Conv2DTranspose(
            num_filters,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="same",
            name=f"{block_prefix}_DecoderBlock{i+1}-TransposedConv",
        )(x)
        x = layers.Activation("relu", name=f"{block_prefix}_DecoderBlock{i+1}-ReLU")(x)

        # All the decoder blocks have concatenate the encoder block output with
        # the same spatial dimentions except for the last one.
        if i != len(decoder_num_filters) - 1:
            x = layers.Concatenate(name=f"{block_prefix}_DecoderBlock{i+1}-Concat")(
                [encoder_blocks_outputs[-(i + 1)], x]
            )

    return x


class ReflectionPadding2D(layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [layers.InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], "REFLECT")


def ck(x, k, use_normalization, use_bias):
    x = layers.Conv2D(
        filters=k, kernel_size=4, strides=2, padding="same", use_bias=use_bias
    )(x)
    if use_normalization:
        x = layers.GroupNormalization(groups=k, center=True, epsilon=1e-5)(
            x, training=True
        )
    x = layers.LeakyReLU(alpha=0.2)(x)
    return x


# First generator layer
def c7Ak(x, k):
    x = layers.Conv2D(
        filters=k, kernel_size=7, strides=1, padding="valid", use_bias=True
    )(x)
    x = layers.GroupNormalization(groups=k, center=True, epsilon=1e-5)(x, training=True)
    x = layers.Activation("relu")(x)
    return x


# Downsampling
def dk(x, k):  # Should have reflection padding
    x = layers.Conv2D(
        filters=k, kernel_size=3, strides=2, padding="same", use_bias=True
    )(x)
    x = layers.GroupNormalization(groups=k, center=True, epsilon=1e-5)(x, training=True)
    x = layers.Activation("relu")(x)
    return x


# Residual block
def Rk(x0, use_dropout=False):
    k = int(x0.shape[-1])

    # First layer
    x = ReflectionPadding2D((1, 1))(x0)
    x = layers.Conv2D(
        filters=k, kernel_size=3, strides=1, padding="valid", use_bias=True
    )(x)
    x = layers.GroupNormalization(groups=k, center=True, epsilon=1e-5)(x, training=True)
    x = layers.Activation("relu")(x)

    if use_dropout:
        x = layers.Dropout(0.5)(x)

    # Second layer
    x = ReflectionPadding2D((1, 1))(x)
    x = layers.Conv2D(
        filters=k, kernel_size=3, strides=1, padding="valid", use_bias=True
    )(x)
    x = layers.GroupNormalization(groups=k, center=True, epsilon=1e-5)(x, training=True)
    # Merge
    x = layers.Add()([x, x0])

    return x


# Upsampling
def uk(x, k, use_resize_convolution=False):
    # (up sampling followed by 1x1 convolution <=> fractional-strided 1/2)
    if use_resize_convolution:
        x = layers.UpSampling2D(size=(2, 2))(x)  # Nearest neighbor upsampling
        x = ReflectionPadding2D((1, 1))(x)
        x = layers.Conv2D(
            filters=k, kernel_size=3, strides=1, padding="valid", use_bias=True
        )(x)
    else:
        x = layers.Conv2DTranspose(
            filters=k, kernel_size=3, strides=2, padding="same", use_bias=True
        )(
            x
        )  # this matches fractionally stided with stride 1/2
    x = layers.GroupNormalization(groups=k, center=True, epsilon=1e-5)(x, training=True)
    x = layers.Activation("relu")(x)
    return x
