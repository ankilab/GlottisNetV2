
DataGenerator
*************
Generation of shuffled and augmented images with coordinates of anterior and posterior points for every batch.

Parameters and output
---------------------

    Parameters: 
        * imgs (DataFrame): First coulmn: image ID, second column: path to image
        * segs (DataFrame): First coulmn: image ID, second column: path to segmentation
        * df_coordinates (DataFrame): z: image ID, ap: anterior point, pp: posterior point
        * batch_size (int): Batch size
        * target_height (int, optional): Target height of images, default = 512 (2D-versions) or 256 (3D-versions)
        * target_width (int, optional): Target width of images, default = 256 (2D-versions) or 128 (3D-versions)
        * augment (bool, optional): If augmentation should be done, default = True
        * shuffle (bool, optional): Shuffling of images, default = True       
        * random (int, optional): Set seed, default=42       
        * radius (except GlottisNetV1) (int, optional): Radius of circle of anterior and posterior points in prediction maps, default = 15 (and GlottisNetV2a-e), 
7.5 (GlottisNetV2 Channels, GlottisNetV2 3DConv, GlottisNetV2 LSTM)

    Returns: 
        * GlottisNetV1:
            * Shape: 4D-tensors (batch, width, height, channels)
                * Augmented and shuffled input images for training of neural network
                * Tuple of segmentation map and 4 coordinates [x (anterior), y (anerior), x(posterior), y(anterior)] for each image in batch
        
        * GlottisNetV2 (a, b, d, e):
            * Shape: 4D-tensors (batch, width, height, channels)
                * Augmented and shuffled input images for training of neural network
                * Tuple of segmentation map, prediction maps (anterior and posterior point stored in 2 channels)  
            
        * GlottisNetV2c
            * Shape: 4D-tensors (batch, width, height, channels)
                * Augmented and shuffled input images for training of neural network
                * Tuple of segmentation map, prediction for anterior point, prediction map for posterior point

        * GlottisNetV2_3DConv and GlottisNetV2_LSTM
            * Shape: 5D-tensors (batch, frames, width, height, channels)
                * Augmented and shuffled input images for training of neural network
                * Tuple of segmentation map and prediction maps (2 channels)

        * GlottisNetV2_Channels
            * Shape: 4D-tensors (batch, width, height, channels)
                * Augmented and shuffled input images (n_frames channels) for training of neural network
                * Tuple of segmentation map and prediction maps (2*n_frames channels)

Helper Functions in DataGenerator
---------------------------------
__len__():
    Calculating number of batches per epoch.

_on_epoch_end():
    Shuffling of training data after each epoch.

_get_augmenter():
    Augmentation of input images.

_create_prediction_map(ap_coord, img_shape, radius):
    Create prediction maps with circle of specific radius at the coordinates of anterior and posterior points.



