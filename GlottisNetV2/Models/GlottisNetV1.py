import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Input, Conv2D, MaxPooling2D, UpSampling2D, Activation, Dense, Concatenate
from tensorflow.keras.layers import concatenate
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D

"""
GlottisNetV1

Parameters
----------

filters : int, optional
    The number of filters in the first layer.
    The subsequent layers have multiples of this filter number. Default: 64

layers : int, optional
    The number of encoding and decoding layers. Default: 4.
    
input_size: (height, width, channels)
    The desired shape of images

Returns
-------

Keras Model
    U-Net structure in tensorflow.keras."""


def glottisnetV1(filters = 64, layers = 4, input_size = (512, 256, 1)):
  
    in_layer = Input(input_size)
    x = in_layer
    
    # save layers to concat
    to_concat = []
    
    # Encoding path
    for step in range(layers):
        
        # First convolution in layer
        x = Conv2D(filters * (2 ** step), (3, 3), use_bias = False, padding = 'same', \
                   strides = 1, kernel_initializer = 'he_uniform')(x)
        
        # Apply batch normalization and activation function
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # Second convolution in layer
        x = Conv2D(filters * (2 ** step), (3, 3), use_bias = False, padding = 'same', \
                   strides = 1, kernel_initializer = 'he_uniform')(x)
        
        # Apply batch normalization and activation function
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # Append input to list to use it for concatenation in decoding path 
        to_concat.append(x)
        x = MaxPooling2D(pool_size = (2, 2))(x)   

    # Last convolution (without MaxPooling) in latent space
    x  = Conv2D(filters * (2 ** (step + 1)), (3, 3), use_bias = False, padding = 'same', strides = 1, \
               kernel_initializer = 'he_uniform')(x)
    
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # ap-point prediction
    out_ap = GlobalAveragePooling2D()(x)
    out_ap = Dense(4, activation = 'relu', use_bias = False, kernel_initializer = 'he_uniform', \
                   name = 'ap_pred')(out_ap)

                  
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
  
    
    # Output segmentation         
    out_seg = Conv2D(1, (1, 1), use_bias = False, padding = "same", activation = 'sigmoid', strides = 1, \
                     kernel_initializer = 'glorot_uniform', name = 'seg')(up_conv)

    
   # Create model
    model = Model(in_layer, [out_ap,  out_seg])
    
    return model
               