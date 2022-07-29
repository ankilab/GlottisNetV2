import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Input, Conv2D, MaxPooling2D, UpSampling2D, Activation, Flatten, Dense, Concatenate, Conv3D, MaxPooling3D, UpSampling3D, ConvLSTM2D
from tensorflow_addons.layers import InstanceNormalization, FilterResponseNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import Sequence
import tensorflow as tf


"""
U-Net with LSTM convolution layers to include temporal information

Parameters
----------

filters : int, optional
    The number of filters in the first layer.
    The subsequent layers have multiples of this filter number. Default: 16

layers : int, optional
    The number of encoding and decoding layers. Default: 4
    
input_size : 4d tuple
    Shape of input tensor. Default: (frames, 512, 256, 1)

Returns
-------

Keras Model
    Using different decoders for prediction maps and segmentation
    U-Net structure in tensorflow.keras with 2 outputs using several frames for training
    First output: Prediction maps of of anterior and posterior point (2 channels)
    Second output: Segmentation map"""


def Decoder(input_tensor, to_concat, name='decoder_', layers=4, filters=16):
    x = input_tensor

    # Decoding path for prediction maps of anterior and posterior points
    for step, filter_factor in enumerate(np.arange(layers)[::-1]):
        
        # First convolution in layer followed by instance normalization and activation function (ReLU)
        x = ConvLSTM2D(filters * (2 ** filter_factor), (3, 3), use_bias=False, padding='same',
                   strides=1, kernel_initializer='he_uniform', name=name + "conv1" + str(step), 
                       return_sequences=True, activation='relu')(x)
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)

        # Umsampling and concatenation with results of encoding path
        x = UpSampling3D(size=(1, 2, 2), name=name + "UpSampling" + str(step))(x)
        x = Concatenate()([x, to_concat[::-1][step]])

    # Last decoding step
    x = Conv3D(filters * (2 ** filter_factor), (3, 3, 3), use_bias=False, padding='same',
               strides=1, kernel_initializer='he_uniform', name=name + "last_decoding_step" + str(step))(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)

    return x


def glottisnetV2e_LSTM(filters=16, layers=4, input_size=(3, 512, 256, 1)):
    in_layer = Input(input_size)
    x = in_layer
    
    # save layers to concat
    to_concat = []

    # Encoding path
    for step in range(layers):
        # Per layer apply two convolutions each followed by batch normalization and an activation function (ReLU)
        x = ConvLSTM2D(filters * (2 ** step), (3, 3), use_bias=False, padding='same', strides=1,
                   kernel_initializer='he_uniform', return_sequences=True, activation='relu')(x)
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)

        # Append input to list to use it for concatenation in decoding path
        to_concat.append(x)
        
        # Contract input
        x = MaxPooling3D(pool_size=(1, 2, 2), padding='same')(x)

    # Last convolution in latent space
    x = ConvLSTM2D(filters * (2 ** (step + 1)), (3, 3), use_bias=False, padding='same', strides=(1, 1),
               kernel_initializer='he_uniform', return_sequences=True, activation='relu')(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)


    # Add two decoders for
    upconv_seg = Decoder(x, to_concat, name="decode_seg", layers = layers, filters=filters)
    upconv_ap = Decoder(x, to_concat, name="decode_ap", layers = layers, filters=filters)

    # Output maps
    # 1x1 convolution to create 3 output maps (segmentation, prediction map of anterior point,
    # prediction map of posterior point)
    out_seg = Conv3D(1, (1, 1, 1), use_bias=False, padding="same", activation='sigmoid', strides=(1, 1, 1),
            kernel_initializer='glorot_uniform', name='seg')(upconv_seg)

    out_ap = Conv3D(2, (1, 1, 1), use_bias=False, padding="same", activation='sigmoid', strides=(1, 1, 1),
                    kernel_initializer='glorot_uniform', name="ap_pred")(upconv_ap)

    # Create model
    model = Model(in_layer, [out_ap, out_seg])

    return model
