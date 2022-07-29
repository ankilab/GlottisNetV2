import cv2
import os
import numpy as np
from skimage.io import imshow
from scipy.ndimage import gaussian_filter
import albumentations as A
from skimage.draw import disk
from tensorflow.keras.utils import Sequence
import random
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

class DataGenerator(Sequence):   
    
    def __init__(self, imgs, segs, batch_size, df_coordinates, target_height = 512, target_width = 256, 
                 shuffle = False, augment = True, random=42):
        
        """Generation of shuffled and augmented images and concerning coordinates of anterior 
            and posterior points for each batch.
        
        Parameters
        ----------
        imgs (Dataframe): Image ID, path to input images       
        
        segs (Dataframe): Image ID, path of  segmentations
        
        df_coordinates : pandas dataframe with coordinates of anterior and posterior points
            z: image_id
            ap: anterior point
            pp: posterior point
        
        batch_size : int
            Batch size for model
        
        target_height: int, optional
            Target height of images, default = 512
        
        target_width: int, optional
            Target width of images, default = 256
        
        augment : bool, optional
            If augmentation should be done, default = True
        
        shuffle : bool, optional
            Shuffling of images, default = True
            
        Returns
        -------
            Augmented and shuffled batch with input images
            Tuple of segmentation map and 4 coordinates (anterior and posterior points) for each image in batch"""               
        
        super().__init__()
        self.imgs = imgs
        self.segs = segs
        self.df_coordinates = df_coordinates
        self.batch_size = batch_size
        self.target_height = target_height
        self.target_width = target_width
        self.shuffle = shuffle
        self.augment = augment
        self.random=random

        # Definition of global variables
        self.on_epoch_end()
        self.aug = self._get_augmenter()         

        
    #Denotes the number of batches per epoch
    def __len__(self):       
        return len(self.imgs) // self.batch_size  
    
    # Prepare next epoch
    def on_epoch_end(self):
        
        # Shuffle images
        if self.shuffle:
            # Rename columns of the dataframe with segmentation to concat the two dataframes
            self.segs=self.segs.rename(columns={'z':'z_seg', 'path': 'path_seg'})
            
            # Concat dataframes
            concat_imgs=pd.concat([self.imgs, self.segs], axis=1)

            # Shuffle obtained dataframe
            concat_imgs=concat_imgs.sample(frac=1, random_state=self.random)
            
            # Write shuffled dataframes back to original dataframe
            self.imgs=concat_imgs[['z', 'path']]
            self.segs=concat_imgs[['z_seg', 'path_seg']]
            
            # Rename columns in dataframe for segmentations
            self.segs=self.segs.rename(columns={'z_seg':'z', 'path_seg': 'path'})

  
    # Data augmentations
    def _get_augmenter(self):      
   
        aug = A.Compose([
            A.RandomBrightnessContrast(p = 0.5),    
            A.Rotate(limit = 30, border_mode = 0, value = [0,0,0], p = 0.75),
            A.HorizontalFlip(p = 0.5),
            A.Blur(p = 0.5),
            #A.GaussNoise(p = 0.5),
            #A.RandomGamma(p=0.75),
            ], keypoint_params=A.KeypointParams(format ='xy', remove_invisible = False))                       
        return aug
    
    
    # Generate one batch of data
    def __getitem__(self, index):
        
        # Lists for input images, output segmentation and prediction maps for 
        # anterior and posterior point
        X = []
        
        # List for output segmentation
        y = []
        
        # List for keypoints of anterior and posterior points (in case return_maps =False)
        ap_coords = []

        # Extract image paths for all images of batch
        X_ids = self.imgs[(index*self.batch_size):(index+1)*self.batch_size]
        Y_ids = self.segs[(index*self.batch_size):(index+1)*self.batch_size]
        
        # Iterate through batch to generate training data
        for i in range(self.batch_size):
            
            # Get image id and construct path of the image and the segmentation
            X_id = X_ids.iloc[i]['z']
            frame_nr = X_ids.iloc[i]['id']
            
            X_path = str(X_ids.iloc[i][1])
            Y_path = str(Y_ids.iloc[i][1])
             
            # Load videos
            cap_img = cv2.VideoCapture(X_path)
            cap_mask = cv2.VideoCapture(Y_path)
            
            vid = []
            mask_vid = []
            
            while(cap_img.isOpened()):
                ret, frame = cap_img.read()
                if ret:
                    vid.append(frame)
                else:
                    break
            cap_img.release()
            
            while(cap_mask.isOpened()):
                ret, frame = cap_mask.read()
                if ret:
                    mask_vid.append(frame)
                else:
                    break
            cap_mask.release()

            # Load frame
            img = vid[frame_nr]
            img = np.asarray(img)
            mask = mask_vid[frame_nr]
            mask = np.asarray(mask)
          
            #Load images
            img = img.astype(np.uint8)
            img_orig = img
            mask = mask.astype(np.uint8)
        
            # Convert RGB-imgages to grayscale image, the channel dimension is deleted
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
            # Convert dataframe wit coordinates of format (2,2) to array of form (4,1) for augmentation of keypoints
            ap_coord = self.df_coordinates[self.df_coordinates['z'] == X_id].iloc[0]["ap"] + \
                       self.df_coordinates[self.df_coordinates['z'] == X_id].iloc[0]["pp"]

            # Precalculate height and width of image
            img_width = img.shape[1]
            img_height = img.shape[0]
                       
            # Resize image if dimensions don't match target dimensions
            if img_height != self.target_height or img_width != self.target_width:
                
                # Resize image and mask
                img = cv2.resize(img, (self.target_width, self.target_height))
                mask = cv2.resize(mask, (self.target_width, self.target_height))
                
                # Adjust coordinates of anterior and posterior point after rezise of image
                ap_coord[0] = ap_coord[0] * (self.target_width / img_width)
                ap_coord[2] = ap_coord[2] * (self.target_width / img_width)
                ap_coord[1] = ap_coord[1] * (self.target_height / img_height)
                ap_coord[3] = ap_coord[3] * (self.target_height / img_height)

            # Bring keypoints to correct shape and pass it to the augmentor
            keypoints = []    
            keypoints.append(tuple(ap_coord[0:2]))
            keypoints.append(tuple(ap_coord[2:4]))  
            
            # Augment image and adjust keypoints accordingly
            if self.augment:
                augmented    = self.aug(image = img, mask = mask, keypoints  = keypoints)
                img          = augmented['image']
                mask         = augmented['mask']
                keypoints    = augmented['keypoints']
            
            # Normalize images and masks
            img = cv2.normalize(img.astype(np.float32),  np.zeros(img.shape), -1, 1, cv2.NORM_MINMAX)
            mask = cv2.normalize(mask.astype(np.float32),  np.zeros(img.shape), 0, 1, cv2.NORM_MINMAX)
            
            X.append(img)
            y.append(np.round(mask))

            ap_coords.append(np.asarray(keypoints[0]+keypoints[1]))
                

        # Convert output lists to numpy arrays and add channel dimension
        # Input: X
        X = np.asarray(X)[..., None]
        
        # Output: Segmentation
        y = np.asarray(y, dtype = np.float32)[..., None]

        ap_coords = np.array(ap_coords)
        return X, (ap_coords, y)
    