import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Input, Conv2D, MaxPooling2D, UpSampling2D, Activation, Dense, Concatenate
from tensorflow_addons.layers import InstanceNormalization, FilterResponseNormalization

"""
GlottisNetV2e (Final version of GlottisNetV2)

Parameters
----------

filters : int, optional
    The number of filters in the first layer.
    The subsequent layers have multiples of this filter number. Default: 16

layers : int, optional
    The number of encoding and decoding layers. Default: 4.

Returns
-------

Keras Model
    Using different decoders for prediction maps of anterior and posterior points and segmentation
    U-Net structure in tensorflow.keras with 2 outputs
    First output: Prediction maps of of anterior and posterior point (2 channels)
    Second output: Segmentation map"""

def Decoder(input_tensor, to_concat, name = 'decoder_', layers = 4, filters = 16):
    x  = input_tensor
        
    # Decoding path for prediction maps of anterior and posterior points
    for step, filter_factor in enumerate(np.arange(layers)[::-1]):
              
        # First convolution in layer followed by instance normalization and activation function
        x = Conv2D(filters * (2 ** filter_factor), (3, 3), use_bias = False, padding = 'same', \
                         strides = 1, kernel_initializer = 'he_uniform', name = name + "conv1" + str(step))(x)
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)
        
        # Umsampling and concatenation with results of encoding path
        x = UpSampling2D(size = (2, 2), name = name + "UpSampling" + str(step))(x)
        x = Concatenate()([x, to_concat[::-1][step]])
        
        # Second convolution followed by batch normalization and activation function
        x = Conv2D(filters * (2 ** filter_factor), (3, 3), use_bias = False, padding = 'same', \
                         strides = 1, kernel_initializer = 'he_uniform', name = name + "conv2" + str(step))(x)
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)

    
    # Last decoding step
    x = Conv2D(filters * (2 ** filter_factor), (3, 3), use_bias = False, padding = 'same', \
                     strides = 1, kernel_initializer = 'he_uniform', name = name + "last_decoding_step" + str(step))(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)

    return x


def glottisnetV2_e(filters = 16, layers = 4, input_size = (512, 256, 1)):
  
    in_layer = Input(input_size)
    x = in_layer
    
    # save layers to concat
    to_concat = []
    
    # Encoding path
    for step in range(layers):
        
        # Per layer apply two convolutions each followed by instance normalization and an activation function
        x = Conv2D(filters * (2 ** step), (3, 3), use_bias = False, padding = 'same', strides = 1, \
                   kernel_initializer = 'he_uniform')(x)
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters * (2 ** step), (3, 3), use_bias = False, padding = 'same', strides = 1, \
                   kernel_initializer = 'he_uniform')(x)
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)

        # Append input to list to use it for concatenation in decoding path 
        to_concat.append(x)
        x = MaxPooling2D(pool_size = (2, 2))(x)   

    # Last convolution (without MaxPooling) in latent space
    x = Conv2D(filters * (2 ** (step + 1)), (3, 3), use_bias = False, padding = 'same', strides = 1, \
               kernel_initializer = 'he_uniform')(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)

    # Add two decoders for
    upconv_seg = Decoder(x, to_concat, name = "decode_seg", layers= layers, filters = filters)
    upconv_ap = Decoder(x, to_concat, name = "decode_ap", layers=layers, filters=filters)
   
    # Output maps          
    # 1x1 convolution to create 3 output maps (segmentation, prediction map of anterior point, prediction map of posterior point)
    out_seg = Conv2D(1, (1, 1), use_bias = False, padding = "same", activation = 'sigmoid', strides = 1, \
                     kernel_initializer = 'glorot_uniform', name = 'seg')(upconv_seg)

    out_ap = Conv2D(2, (1, 1), use_bias = False, padding = "same", activation = 'sigmoid', strides = 1, \
                    kernel_initializer ='glorot_uniform', name = 'ap_pred')(upconv_ap)

    
   # Create model
    model = Model(in_layer, [out_ap, out_seg])
    
    return model
               