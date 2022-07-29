import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Input, Conv2D, MaxPooling2D, UpSampling2D, Activation, Dense, Concatenate
from tensorflow.keras.layers import concatenate

"""
GlottisNetV2d

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
    The last layer in decoding path is separated for the segmentation of the glottal area and the anterior and posterior point prediction
    U-Net structure in tensorflow.keras with 2 outputs
    First output: Prediction maps of of anterior and posterior point (2 channels)
    Second output: Segmentation map"""

def Decoder(input_tensor, to_concat, layers = 4, filters = 64):
    x  = input_tensor
        
    # Decoding path for prediction maps of anterior and posterior points
    for step, filter_factor in enumerate(np.arange(1, layers)[::-1]):       
        
        # First convolution in layer followed by batch normalization and activation function
        x = Conv2D(filters * (2 ** filter_factor), (3, 3), use_bias = False, padding = 'same', \
                         strides = 1, kernel_initializer = 'he_uniform')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # Umsampling and concatenation with results of encoding path
        x = UpSampling2D(size = (2, 2))(x)
        x = Concatenate()([x, to_concat[::-1][step]])
        
        # Second convolution followed by batch normalization and activation function
        x = Conv2D(filters * (2 ** filter_factor), (3, 3), use_bias = False, padding = 'same', \
                         strides = 1, kernel_initializer = 'he_uniform')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
          
    ######## AP output
    # First convolution followed by batch normalization and activation function
    x1 = Conv2D(filters * (2 ** 0), (3, 3), use_bias = False, padding = 'same', \
                     strides = 1, kernel_initializer = 'he_uniform')(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    # Umsampling and concatenation with results of encoding path
    x1 = UpSampling2D(size = (2, 2))(x1)
    x1 = Concatenate()([x1, to_concat[::-1][3]])

    # Second convolution followed by batch normalization and activation function
    x1 = Conv2D(filters * (2 ** 0), (3, 3), use_bias = False, padding = 'same', \
                     strides = 1, kernel_initializer = 'he_uniform')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)  
    
    # Last decoding step
    x1 = Conv2D(filters * (2 ** 0), (3, 3), use_bias = False, padding = 'same', \
                     strides = 1, kernel_initializer = 'he_uniform')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    
     
    ######## Segmentation output
    # First convolution followed by batch normalization and activation function
    x2 = Conv2D(filters * (2 ** 0), (3, 3), use_bias = False, padding = 'same', \
                     strides = 1, kernel_initializer = 'he_uniform')(x)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)

    # Umsampling and concatenation with results of encoding path
    x2 = UpSampling2D(size = (2, 2))(x2)
    x2 = Concatenate()([x2, to_concat[::-1][3]])

    # Second convolution followed by batch normalization and activation function
    x2 = Conv2D(filters * (2 ** 0), (3, 3), use_bias = False, padding = 'same', \
                     strides = 1, kernel_initializer = 'he_uniform')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    
    # Last decoding step
    x2 = Conv2D(filters * (2 ** 0), (3, 3), use_bias = False, padding = 'same', \
                     strides = 1, kernel_initializer = 'he_uniform')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)

    return x1, x2

def glottisnetV2_d(filters = 64, layers = 4, input_size = (512, 256, 1)):
  
    in_layer = Input(input_size)
    x = in_layer
    
    # Save layers to concat
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

    # Decoder
    # The last layer of the decoder is separated for segmenting the glottal area and predicting anterior and posterior points
    upconv_ap, upconv_seg= Decoder(x, to_concat, filters = filters, layers = layers)   
    
    # Output maps          
    # 1x1 convolution to create 3 output maps (segmentation, prediction map of anterior point, prediction map of posterior point)
    out_seg = Conv2D(1, (1, 1), use_bias = False, padding = "same", activation = 'sigmoid', strides = 1, \
                     kernel_initializer = 'glorot_uniform', name = 'seg')(upconv_ap)

    out_ap = Conv2D(2, (1, 1), use_bias = False, padding = "same", activation = 'sigmoid', strides = 1, \
                    kernel_initializer ='glorot_uniform', name = 'ap_pred')(upconv_seg)
    
    # Create model
    model = Model(in_layer, [out_ap, out_seg])
    
    return model
               