import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Input, Conv2D, MaxPooling2D, UpSampling2D, Activation, Flatten, Dense, Concatenate
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import Sequence

"""
Building a U-Net


Parameters
----------

filters : int, optional
    The number of filters in the first layer.
    The subsequent layers have multiples of this filter number. Default: 64

layers : int, optional
    The number of encoding and decoding layers. Default: 4.

Returns
-------

Keras Model
    U-Net structure in tensorflow.keras.
    3 Outputs
    1. Prediction map of anterior point
    2. Prediction map of posterior point
    3. Segmentation"""


def glottisnetV2_c(filters = 64, layers = 4, input_size = (512, 256, 1)):
  
    in_layer = Input(input_size)
    x = in_layer
    
    # save layers to concat
    to_concat = []
    
    # Encoding path
    for step in range(layers):
        
        # Per layer apply two convolutions each followed by batch normalization and an activation function
        x = Conv2D(filters * (2 ** step), (3, 3), use_bias = False, padding = 'same', strides = 1, \
                   kernel_initializer = 'he_uniform')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters * (2 ** step), (3, 3), use_bias = False, padding = 'same', strides = 1, \
                   kernel_initializer = 'he_uniform')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # Append input to list to use it for concatenation in decoding path 
        to_concat.append(x)
        x = MaxPooling2D(pool_size = (2, 2))(x)   

    # Last convolution (without MaxPooling) in latent space
    x = Conv2D(filters * (2 ** (step + 1)), (3, 3), use_bias = False, padding = 'same', strides = 1, \
               kernel_initializer = 'he_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)  
               
    up_conv = x

    # Decoding path
    for step, filter_factor in enumerate(np.arange(layers)[::-1]):
        
        
        # First convolution in layer followed by batch normalization and activation function
        up_conv = Conv2D(filters * (2 ** filter_factor), (3, 3), use_bias = False, padding = 'same', \
                         strides = 1, kernel_initializer = 'he_uniform')(up_conv)
        up_conv = BatchNormalization()(up_conv)
        up_conv = Activation('relu')(up_conv)

        
        # Umsampling and concatenation with results of encoding path
        up_conv = UpSampling2D(size = (2, 2))(up_conv)
        up_conv = Concatenate()([up_conv, to_concat[::-1][step]])

        
        # Second convolution followed by batch normalization and activation function
        up_conv = Conv2D(filters * (2 ** filter_factor), (3, 3), use_bias = False, padding = 'same', \
                         strides = 1, kernel_initializer = 'he_uniform')(up_conv)
        up_conv = BatchNormalization()(up_conv)
        up_conv = Activation('relu')(up_conv)

    
    # Last decoding step
    up_conv = Conv2D(filters * (2 ** filter_factor), (3, 3), use_bias = False, padding = 'same', \
                     strides = 1, kernel_initializer = 'he_uniform')(up_conv)
    up_conv = BatchNormalization()(up_conv)
    up_conv = Activation('relu')(up_conv)
    
    # Output maps          
    # 1x1 convolution to create 3 output maps (segmentation, prediction map of anterior point, prediction map of posterior point)
    # First ouput: perdiction map of anterior point
    # Second output: prediction map of posterior point
    # Third output: Segmentation map
    out_seg = Conv2D(1, (1, 1), use_bias = False, padding = "same", activation = 'sigmoid', strides = 1, \
                     kernel_initializer = 'glorot_uniform', name = 'seg')(up_conv)

    out_ap = Conv2D(1, (1, 1), use_bias = False, padding = "same", activation = 'sigmoid', strides = 1, \
                    kernel_initializer ='glorot_uniform', name = 'a_point')(up_conv)

    out_pp = Conv2D(1, (1, 1), use_bias = False, padding = "same", activation = 'sigmoid', strides = 1, \
                    kernel_initializer = 'glorot_uniform', name = 'p_point')(up_conv)

    
   # Create model
    model = Model(in_layer, [out_ap, out_pp,  out_seg])
    
    return model
               