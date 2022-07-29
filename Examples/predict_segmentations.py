import numpy as np
import flammkuchen as fl
import cv2
import os
from tqdm.notebook import tqdm
import imageio as io
import tensorflow as tf
from tensorflow.keras.models import load_model
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import tensorflow_addons as tfa
from GlottisNetV2.Utils.data import load_data

# Iterate through videos
for vi in range(0, 640):

    # Set path of current video
    vpath = r"Fill in path" + str(vi) + ".mp4" # TODO: Set path

    if os.path.exists(vpath):

        # Set model path
        path_model = r"" # TODO: Set path

        # Load frames fo video
        ims = io.mimread(vpath, memtest=False)

        # Load model
        Unet = load_model(path_model, compile=False,
                          custom_objects={'InstanceNormalization': tfa.layers.InstanceNormalization})

        # Initialize lists to hold data
        masks = []
        ims_orig = []

        # Iterate through frames
        for i in range(len(ims)):

            # Preprocess image for prediction
            img_orig = ims[i].astype(np.float32)

            # Color --> gray and normalize image
            img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img_orig, (256, 512))

            # Normalize and preprocess image
            normalizedImg = np.zeros(img.shape)
            img = cv2.normalize(img, normalizedImg, -1, 1, cv2.NORM_MINMAX)
            img = img[None, ..., None]

            # Prediction
            pred_maps, seg_pred = Unet.predict(img)
            mask = np.asarray(np.squeeze(seg_pred))

            # Convert probabilities to boolean
            mask = np.round(mask)

            # Resize, convert and transpose mask to get the right shape [frames, x, y] (type: boolean)
            mask = cv2.resize(mask, (img_orig.shape[1], img_orig.shape[0]))
            mask = mask.astype(bool)
            mask = np.transpose(mask, (1, 0))
            img_orig = np.transpose(img_orig, (1, 0))

            # Append images to lists
            masks.append(mask)
            ims_orig.append(img_orig)

        # Convert list to numpy array
        masks = np.asarray(masks)

        # Save sequence of masks as .mask file
        # path1 = r"Fill in path" + str(vi) + ".mask" # TODO: Set path
        # fl.save(path1, {"mask": masks, 'files': ims_orig}, compression='blosc')










