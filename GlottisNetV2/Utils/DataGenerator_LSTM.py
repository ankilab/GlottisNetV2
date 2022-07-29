import cv2
import os
import numpy as np
from skimage.io import imshow
import albumentations as A
from skimage.draw import disk
from tensorflow.keras.utils import Sequence
import random
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import imageio as io


class DataGenerator(Sequence):

    def __init__(self, imgs, segs, batch_size, df_coordinates, target_height=512, target_width=256, frame_nr=3,
                 shuffle=False, augment=True, radius=7.5, random = 42):

        """Generation of shuffled and augmented images with coordinates of anterior and posterior points for every batch..

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
            
        random : int, optional
            Set seed, default=42
            
        radius : int, optional
            Radius of circle of anterior and posterior points in prediction maps, default=7.5


        Returns
        -------
            5D-tensors (batch, frames, width, height, channels)
            Augmented and shuffled input image
            Tuple of segmentation map and prediction maps (2 channels)"""
        
        
        super().__init__()
        self.imgs = imgs
        self.segs = segs
        self.df_coordinates = df_coordinates
        self.batch_size = batch_size
        self.target_height = target_height
        self.target_width = target_width
        self.shuffle = shuffle
        self.augment = augment
        self.random = random
        self.frame_nr = frame_nr
        self.radius = radius

        # Definition of global variables
        self.on_epoch_end()
        self.aug = self._get_augmenter()

    # Denotes the number of batches per epoch
    def __len__(self):
        return len(self.imgs) // self.batch_size

    # Prepare next epoch
    def on_epoch_end(self):

        # Shuffle images
        if self.shuffle:
            
            # Rename columns of the dataframe with segmentation to concat the two dataframes
            self.segs = self.segs.rename(columns={'z': 'z_seg', 'path': 'path_seg', 'id': 'id_seg'})
            
            # Concat dataframes
            concat_imgs = pd.concat([self.imgs, self.segs], axis=1)

            # Shuffle obtained dataframe
            concat_imgs = concat_imgs.sample(frac=1, random_state=self.random)
            
            # Write shuffled dataframes back to original dataframe
            self.imgs = concat_imgs[['z', 'path', 'id']]
            self.segs = concat_imgs[['z_seg', 'path_seg', 'id_seg']]
            
            # Rename columns in dataframe for segmentations
            self.segs = self.segs.rename(columns={'z_seg': 'z', 'path_seg': 'path', 'id_seg': 'id'})

    # Data augmentations
    def _get_augmenter(self):
        
        aug = A.Compose(
            [A.RandomBrightnessContrast(p=0.5),
             A.Rotate(limit=30, border_mode=0, value=[0, 0, 0], p=0.75),
             A.Blur(p=0.5),
             A.GaussNoise(p=0.5),
             A.RandomGamma(p=0.75)],
            
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
            additional_targets= {'image1': 'image', 'image2': 'image', 
                                 'image3': 'image', 'image4': 'image', 'image5': 'image',
                                 'image6': 'image', 'image7': 'image', 'image8': 'image', 
                                 'image9': 'image', 'image10': 'image', 'image11': 'image',
                                 'mask1': 'mask', 'mask2': 'mask',
                                 'mask3': 'mask', 'mask4': 'mask', 'mask5': 'mask',
                                 'mask6': 'mask', 'mask7': 'mask', 'mask8': 'mask',
                                 'mask9': 'mask', 'mask10': 'mask', 'mask11': 'mask'},
            p=1)
        return aug

    # Create prediction map
    def _create_prediction_map(self, ap_coord, img_shape, radius=7.5):

        # Initialize prediction map
        pred_map = np.zeros(img_shape)

        # Create two circles with its center at the coordinates of thr anterior and posterior point
        x_ap, y_ap = disk([ap_coord[1], ap_coord[0]], radius, shape=img_shape)
        # Set previously defined circles to one in the prediction map
        pred_map[x_ap, y_ap] = 1

        return pred_map

    # Generate one batch of data
    def __getitem__(self, index):

        # Lists for input videos
        X = []

        # List for output segmentations
        y = []

        # List for prediction maps of anterior and posterior points of glottis in videos
        y_ap = []
        y_pp = []

        # List for keypoints of anterior and posterior points
        ap_coords = []

        # Extract paths for all videos of batch
        X_ids = self.imgs[(index * self.batch_size):(index + 1) * self.batch_size]
        Y_ids = self.segs[(index * self.batch_size):(index + 1) * self.batch_size]

        # Iterate through batch to generate training data
        for i in range(self.batch_size):

            # Get video id and construct path of the video and the segmentation
            X_id = X_ids.iloc[i][0]

            X_path = str(X_ids.iloc[i][1])
            Y_path = str(Y_ids.iloc[i][1])
            img = []
            mask = []
            cap_img = cv2.VideoCapture(X_path)
            cap_mask = cv2.VideoCapture(Y_path)
            
            while(cap_img.isOpened()):
                ret, frame = cap_img.read()
                if ret:
                    img.append(frame)
                else:
                    break
            cap_img.release()
            
            while(cap_mask.isOpened()):
                ret, frame = cap_mask.read()
                if ret:
                    mask.append(frame)
                else:
                    break
            cap_mask.release()

            # Extract n frames from video
            nr = X_ids.iloc[i][2]
            start = nr-self.frame_nr
            end = nr
            imgs = img[start:end]
            masks = mask[start:end]
            imgs = np.asarray(imgs)
            masks = np.asarray(masks)

            # Convert RGB-images to grayscale image, the channel dimension is deleted
            frames = np.zeros((imgs.shape[0], imgs.shape[1], imgs.shape[2]), dtype=np.uint8)
            frames_seg = np.zeros((imgs.shape[0], imgs.shape[1], imgs.shape[2]), dtype=np.uint8)
            
            # Convert RGB - images to grayscale image, the channel dimension is deleted
            for frame in range(len(imgs)):
                fr=imgs[frame].astype(np.uint8)
                fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                frames[frame, :, :] = fr

            for frame_seg in range(len(masks)):
                fr=masks[frame_seg].astype(np.uint8)
                fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                frames_seg[frame_seg, :, :] = fr

            # Convert dataframe wit coordinates of format (2,2) to array of form (4,1) for augmentation of keypoints
            ap_coord = self.df_coordinates[self.df_coordinates['z'] == X_id].iloc[0]["ap"] + \
                       self.df_coordinates[self.df_coordinates['z'] == X_id].iloc[0]["pp"]

            # Precalculate height and width of image
            img_width = imgs.shape[2]
            img_height = imgs.shape[1]

            # Buffer for resized frames
            new_fr = np.zeros((imgs.shape[0], self.target_height, self.target_width), dtype=np.uint8)
            new_fr_seg = np.zeros((imgs.shape[0], self.target_height, self.target_width), dtype=np.uint8)

            # Resize frames if their dimensions don't match the target dimensions
            if img_height != self.target_height or img_width != self.target_width:

                # Iterate through all frames of video and resize them
                for frame_nr in range(len(imgs)):

                    # Resize image and mask
                    fr = frames[frame_nr, :, :]
                    fr = cv2.resize(fr, (self.target_width, self.target_height))                  
                    new_fr[frame_nr] = fr
                    
                    fr_seg = frames_seg[frame_nr, :, :]
                    fr_seg = cv2.resize(fr_seg, (self.target_width, self.target_height))
                    new_fr_seg[frame_nr] = fr_seg
                    
                # Adjust coordinates of anterior and posterior point after resize of frame
                ap_coord[0] = ap_coord[0] * (self.target_width / img_width)
                ap_coord[2] = ap_coord[2] * (self.target_width / img_width)
                ap_coord[1] = ap_coord[1] * (self.target_height / img_height)
                ap_coord[3] = ap_coord[3] * (self.target_height / img_height)        

                frames = new_fr
                frames_seg = new_fr_seg
            
            # Bring keypoints to correct shape and pass it to the augmentor
            keypoints = []
            keypoints.append(tuple(ap_coord[0:2]))
            keypoints.append(tuple(ap_coord[2:4]))

            # Augment image and adjust keypoints accordingly
            if self.augment:
                augmented = self.aug(image=frames[0, :, :], image1=frames[1, :, :], image2=frames[2, :, :],
                                     image3=frames[3, :, :], image4=frames[4, :, :], image5=frames[5, :, :],
                                     image6=frames[6, :, :], image7=frames[7, :, :], image8=frames[8, :, :],
                                     image9=frames[9, :, :], image10=frames[10, :, :], image11=frames[11, :, :],
                                     mask=frames_seg[0, :, :], mask1=frames_seg[1, :, :], mask2=frames_seg[2, :, :],
                                     mask3=frames_seg[3, :, :], mask4=frames_seg[4, :, :], mask5=frames_seg[5, :, :],
                                     mask6=frames_seg[6, :, :], mask7=frames_seg[7, :, :], mask8=frames_seg[8, :, :],
                                     mask9=frames_seg[9, :, :], mask10=frames_seg[10, :, :], mask11=frames_seg[11, :, :],
                                     keypoints=keypoints)
                
                frames[0, :, :] = augmented['image']
                frames[1, :, :] = augmented['image1']
                frames[2, :, :] = augmented['image2']
                frames[3, :, :] = augmented['image3']
                frames[4, :, :] = augmented['image4']
                frames[5, :, :] = augmented['image5']
                frames[6, :, :] = augmented['image6']
                frames[7, :, :] = augmented['image7']
                frames[8, :, :] = augmented['image8']
                frames[9, :, :] = augmented['image9']
                frames[10, :, :] = augmented['image10']
                frames[11, :, :] = augmented['image11']

                frames_seg[0, :, :] = augmented['mask']
                frames_seg[1, :, :] = augmented['mask1']
                frames_seg[2, :, :] = augmented['mask2']
                frames_seg[3, :, :] = augmented['mask3']
                frames_seg[4, :, :] = augmented['mask4']
                frames_seg[5, :, :] = augmented['mask5']
                frames_seg[6, :, :] = augmented['mask6']
                frames_seg[7, :, :] = augmented['mask7']
                frames_seg[8, :, :] = augmented['mask8']
                frames_seg[9, :, :] = augmented['mask9']
                frames_seg[10, :, :] = augmented['mask10']
                frames_seg[11, :, :] = augmented['mask11']
                
                keypoints = augmented['keypoints']
            
            new_fr1 = np.zeros((imgs.shape[0], self.target_height, self.target_width), dtype=np.float32)
            new_seg1 = np.zeros((imgs.shape[0], self.target_height, self.target_width), dtype=np.float32)
            
            # Iterate through all frames of video and normalize them
            for frame_nr in range(len(imgs)):
                # Normalize images and masks
                img = cv2.normalize(frames[frame_nr, :, :].astype(np.float32),
                                    np.zeros((frames[frame_nr, :, :].astype(np.float32)).shape),
                                    -1, 1, cv2.NORM_MINMAX)
                mask = cv2.normalize(frames_seg[frame_nr, :, :].astype(np.float32),
                                     np.zeros((frames[frame_nr, :, :].astype(np.float32)).shape),
                                     0, 1, cv2.NORM_MINMAX)

                new_fr1[frame_nr, :, :] = img
                new_seg1[frame_nr, :, :] = np.round(mask)
                
            frames = new_fr1
            frames_seg = new_seg1

            # Append original video and segmentation to list
            X.append(frames)
            y.append(frames_seg)

            # Create prediction maps
            map_ap = self._create_prediction_map(keypoints[0], img.shape, self.radius)
            map_pp = self._create_prediction_map(keypoints[1], img.shape, self.radius)

            map_ap2 = np.asarray(np.zeros((self.frame_nr, img.shape[0], img.shape[1])))
            map_pp2 = np.asarray(np.zeros((self.frame_nr, img.shape[0], img.shape[1])))

            for k in range(self.frame_nr):
                map_ap2[k, :, :] = map_ap
                map_pp2[k, :, :] = map_pp

            # Append images and labels to list and return the dataset
            y_ap.append(map_ap2)
            y_pp.append(map_pp2)

        # Convert output lists to numpy arrays and add channel dimension
        # Input: X
        X = np.asarray(X)[..., None]

        # Output: Segmentation)
        y = np.asarray(y, dtype=np.float32)[..., None]

        # Output: Prediction map for anterior point
        y_ap = np.asarray(y_ap, dtype=np.float32)[..., None]

        # Output: Prediction map for posterior point
        y_pp = np.array(y_pp, dtype=np.float32)[..., None]

        # Two exits
        # First exit: predictions of anterior and posterior points (2 channels)
        # Second exit: Segmentations
        y_keypoints = np.append(y_ap, y_pp, axis=-1)
        return X, [y_keypoints, y]


