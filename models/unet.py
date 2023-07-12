import tensorflow as tf
import keras
from keras import layers
from einops.layers.tensorflow import Rearrange

from typing import List, Tuple


def UNet(
    input_shape: Tuple[int, int, int],
    init_dim: int,
    dim: int,
    dim_mults: List[int],
    groups: int = 8,
    out_channels: int = 3,
    out_activation: str = "sigmoid",
):
    dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
    in_out = list(zip(dims[:-1], dims[1:]))

    h = []
    inputs = keras.Input(input_shape)
    x = r = layers.Conv2D(init_dim, kernel_size=3, padding="same")(inputs)

    # Encoder.
    for i, (dim_in, dim_out) in enumerate(in_out):
        is_last = i >= len(in_out) - 1

        x = ResnetBlock(dim_in, groups=groups)(x)
        h.append(x)
        x = ResnetBlock(dim_in, groups=groups)(x)
        x = Residual(PreNorm(LinearAttention(dim_in)))(x)
        h.append(x)

        x = (
            Downsample(dim_out)(x)
            if not is_last
            else layers.Conv2D(dim_out, 3, padding="same")(x)
        )

    # Middle/bottleneck.
    x = ResnetBlock(dims[-1], groups=groups)(x)
    x = Residual(PreNorm(Attention(dims[-1])))(x)
    x = ResnetBlock(dims[-1], groups=groups)(x)

    # Decoder.
    for i, (dim_in, dim_out) in enumerate(reversed(in_out)):
        is_last = i == (len(in_out) - 1)

        x = layers.Concatenate()([x, h.pop()])
        x = ResnetBlock(dim_out, groups=groups)(x)

        x = layers.Concatenate()([x, h.pop()])
        x = ResnetBlock(dim_out, groups=groups)(x)
        x = Residual(PreNorm(LinearAttention(dim_out)))(x)

        x = (
            Upsample(dim_in)(x)
            if not is_last
            else layers.Conv2D(dim_in, 3, padding="same")(x)
        )

    x = layers.Concatenate()([x, r])
    x = ResnetBlock(init_dim, groups=groups)(x)
    x = layers.Conv2D(out_channels, kernel_size=1)(x)
    x = layers.Activation(out_activation)(x)
    return keras.Model(inputs=inputs, outputs=x)


# ---- Main layers ----


def ResnetBlock(dim: int, groups: int = 8):
    def apply(inputs):
        same_dim = dim == inputs.shape[-1]
        residual_layer = (
            layers.Identity() if same_dim else layers.Conv2D(dim, kernel_size=1)
        )

        x = Block(dim, groups=groups)(inputs)
        x = Block(dim, groups=groups)(x)
        x = layers.Add()([x, residual_layer(inputs)])
        return x

    return apply


def LinearAttention(dim: int, heads: int = 4, dim_head: int = 32):
    """Took from Efficient Attention: Attention with Linear Complexities paper
    https://arxiv.org/pdf/1812.01243.pdf"""
    scale = dim_head**-0.5

    def apply(inputs):
        b, h, w, c = inputs.shape

        qkv = layers.Conv2D(
            dim_head * heads * 3, kernel_size=1, padding="same", use_bias=False
        )(inputs)
        qkv = tf.split(qkv, 3, axis=-1)
        q, k, v = map(
            lambda t: Rearrange("b x y (h c) -> b h c (x y)", h=heads)(t), qkv
        )
        q = tf.nn.softmax(q, axis=-2) * scale
        k = tf.nn.softmax(k, axis=-1)

        context = tf.einsum("b h d n, b h e n -> b h d e", k, v)
        out = tf.einsum("b h d e, b h d n -> b h e n", context, q)
        out = Rearrange("b h c (x y) -> b x y (h c)", h=heads, x=h, y=w)(out)
        out = layers.Conv2D(dim, kernel_size=1, padding="same")(out)
        out = layers.GroupNormalization(groups=1)(out)

        return out

    return apply


def Attention(dim: int, heads: int = 4, dim_head: int = 32):
    scale = dim_head**-0.5
    hidden_dim = dim_head * heads

    def apply(inputs):
        b, h, w, c = inputs.shape
        qkv = layers.Conv2D(
            hidden_dim * 3, kernel_size=1, padding="same", use_bias=False
        )(inputs)
        qkv = tf.split(qkv, 3, axis=-1)
        q, k, v = map(
            lambda t: Rearrange("b x y (h c) -> b h c (x y)", h=heads)(t), qkv
        )
        q = q * scale

        sim = tf.einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - tf.reduce_max(sim, axis=-1, keepdims=True)
        attn = tf.nn.softmax(sim, axis=-1)

        out = tf.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = Rearrange("b h (x y) d -> b x y (h d)", x=h, y=w)(out)
        out = layers.Conv2D(dim, kernel_size=1, padding="same")(out)

        return out

    return apply


def Residual(fn):
    def apply(inputs):
        x = fn(inputs)
        return layers.Add()([x, inputs])

    return apply


def PreNorm(fn):
    def apply(inputs):
        x = layers.GroupNormalization(1)(inputs)
        x = fn(x)
        return x

    return apply


def Upsample(dim: int):
    def apply(inputs):
        x = layers.UpSampling2D()(inputs)
        x = layers.Conv2D(dim, kernel_size=3, padding="same")(x)
        return x

    return apply


def Downsample(dim: int):
    def apply(inputs):
        x = layers.AveragePooling2D()(inputs)
        x = layers.Conv2D(dim, kernel_size=3, padding="same")(x)
        return x

    return apply


def Block(dim: int, groups: int = 8):
    def apply(inputs):
        x = layers.Conv2D(dim, kernel_size=3, padding="same")(inputs)
        x = layers.GroupNormalization(groups)(x)
        x = layers.Activation("swish")(x)
        return x

    return apply
