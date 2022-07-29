import os
import pandas as pd
import json
import cv2
import numpy as np
from tqdm.notebook import tqdm
import tensorflow as tf

'''
Load anterior and posterior point to dataframe. 

Parameters
----------
aplist: JSON-file
    Contains anterior and posterior points of each image.
n: number of training images

Returns
-------
DataFrame with three columns: image id, coordinates of anterior and coordinates of posterior points.
'''

def load_data(aplist, n):

    # Load jason files
    with open(aplist) as ap_file:
        df_ap = json.load(ap_file)

    # Convert data from json file to pd.dataframe
    df_ap = pd.DataFrame(df_ap['rois'])
    
    # Sort dataframe
    df_ap = df_ap.sort_values('z', axis = 0)
    
    # Extract n samples
    df_ap = df_ap[:n * 2]

    # Set column names
    cols = ['z', 'ap', 'pp']
    
    # Create new pandas dataframe for rearranged coordinates of anterior and posterior points
    df_ap_new = pd.DataFrame(columns=cols)
    
    # Rearrange dataframe to the form [z, x[p,a], y[p,a]]
    # Iterate through dataframe
    
    # Make sure to run this for-loop only once by saving the last image name and comparing it.
    save_z = -1
    for row in tqdm(range(len(df_ap))):    
        
        # Extract id of image
        img_z = df_ap.iloc[row]['z']
        
        # When current row has a different id than the last id
        if df_ap.iloc[row]['z'] != save_z:
            
            # Find all images with the same image number as "z" (normally 2).                 
            rows_cond = df_ap.loc[df_ap['z'] == img_z]

            # Sort obtained dataframe
            rows_cond = rows_cond.sort_values('id', axis = 0)

            # Check if coordinates are missing.
            # Create a row for the new data frame containing rearranged coordinates.

            if len(rows_cond) == 1:

                # Use [0,0] for missing coordinates
                
                print('One coordinate is missing in image: ', rows_cond.iloc[0]['z'])

                # Replace anterior point by [0,0]
                if df_ap.iloc[row]['id'] == 0:
                    new_row = {'z': df_ap.iloc[row]['z'], 'ap': [1000, 1000], 'pp': rows_cond.iloc[0]['pos']}

                # Replace posterior point by [0,0]
                else:
                    new_row = {'z': df_ap.iloc[row]['z'], 'ap': rows_cond.iloc[0]['pos'], 'pp': [1000, 1000]}

            # Anterior and posterior points are inside image. No coordinate is missing.
            else:
                # Create a row of new dataframe containing the the coordinates of anterior
                # and posterior point [pp, ap]
                new_row = {'z': df_ap.iloc[row]['z'], 'ap': rows_cond.iloc[1]['pos'], \
                                    'pp': rows_cond.iloc[0]['pos']}

            # Append row to new dataframe
            df_ap_new = df_ap_new.append(new_row, ignore_index = True)
        save_z = df_ap.iloc[row]['z']
     
    return df_ap_new
    

def load_img(path, target_height = 512, target_width = 256):
    
    # Load image
    img = cv2.imread(path).astype(np.float32)

    # color --> gray and normalize image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (target_width, target_height))

    # normalize and preprocess image
    normalizedImg = np.zeros(img.shape)
    img = cv2.normalize(img,  normalizedImg, -1, 1, cv2.NORM_MINMAX)

    return img[None, ..., None]



'''Calculate mass of input image
   
   Parameters
   ----------
   img: Prediction map of anterior and posterior point
   
   Returns
   -------
   Coordinates of mass of input image'''

def img_moment(img):
    
    # Convert tensor to numpy array
    img=np.asarray(img)
    
    # Calculate moments
    M = cv2.moments(img)
    
    # Calculate centroids
    # Catch the case when M0 is NaN or 0
    if not np.isnan(M['m00']) and M['m00']>0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
    else:
        cx = 1
        cy = 1
    return (cx, cy)



'''Calculate MAPE
   
   Parameters
   ----------
   keypoints_orig: Coordinates of original keypoints
   keypoints_pred: Coordinates of predicted keypoints
   
   Returns
   -------
   MAPE'''

def MAPE(keypoints_orig, keypoints_pred, nr_points, eps = 1e-9):
    
    # Initialize list for MAPES of all items in batch
    mapes = []      
    keypoints_orig = keypoints_orig.copy()
    # Iterate over keypoints
    for i in range(len(keypoints_orig)):
        
        # Check if the reference image is empty. 
        if all(j >= 0 for j in keypoints_orig[i]):

            # Calculate MAPE of original and predicted keypoints
            keypoints_orig[i][keypoints_orig[i] == 0.0] = 1.0

            tmp = np.abs((keypoints_orig[i] - keypoints_pred[i]) / (keypoints_orig[i]+eps))
            mape = (np.sum(tmp)) * (100 / nr_points)

            # Append calculated MAPE score to batch
            mapes.append(mape)
        else:
            mapes.append(-1.0)          
    return mapes


'''Calculate MAPE of several frames
   
   Parameters
   ----------
   keypoints_orig: Coordinates of original keypoints
   keypoints_pred: Coordinates of predicted keypoints
   
   Returns
   -------
   MAPE'''

def MAPE3D(keypoints_orig, keypoints_pred, nr_points, eps = 1e-9):
    
    # Initialize list for MAPES of all items in batch   
    mapes_batch=[]
    
    # Iterate over keypoints
    for i in range(len(keypoints_orig)):
        
        mapes = []   
        for j in range(len(keypoints_orig[0])):
        
            # Check if the reference image is empty. 
            if all(j >= 0 for j in keypoints_orig[i][j]):

                keypoints_orig_new = keypoints_orig[i][j]
                keypoints_pred_new = keypoints_pred[i][j]
                
                # Calculate MAPE of original and predicted keypoints
                keypoints_orig_new[keypoints_orig_new == 0.0] = 1.0

                tmp = np.abs((keypoints_orig_new - keypoints_pred_new) / (keypoints_orig_new+eps))
                mape = (np.sum(tmp)) * (100 / nr_points)

                # Append calculated MAPE score to batch
                mapes.append(mape)
            else:
                mapes.append(-1.0)
                
        mapes_batch.append(np.mean(mapes))
    return mapes_batch


'''Metric to evaluate MAPE of several frames
   
   Parameters
   ----------
   y_true: Coordinates of original keypoints --> (1,4)
   y_pred: Coordinates of predicted keypoints --> (1,4)
   
   Returns
   -------
   MAPE'''

def metric_mape3D(y_true, y_pred):
    
    # Initialize lists for original and predicted keypoints
    keypoints_true = []
    keypoints_pred = []
    
    # GlottisNetV2 3DConv and GlottisnetV2 LSTM
    if np.ndim(y_pred) == 5:      
        
        for img_nr in range(y_pred.shape[0]):
            ap_ref = np.zeros((y_true.shape[1], 2))
            pp_ref = np.zeros((y_true.shape[1], 2))
            ap_pred = np.zeros((y_true.shape[1], 2))
            pp_pred = np.zeros((y_true.shape[1], 2))
            
            for frame_nr in range(y_pred.shape[1]):
                
                # If reference image of prediction map of anterior point is not empty, calculate image moment
                # else write -1 to keypoint list to recognize it later
                
                if np.sum(y_true[img_nr, frame_nr, :, :, 0]) > 0:
                    ap_ref[frame_nr, :] = img_moment(y_true[img_nr, frame_nr, :, :, 0])
                    ap_pred[frame_nr, :] = img_moment(y_pred[img_nr, frame_nr, :, :, 0])
                else:
                    ap_ref[frame_nr, :] = [-1.0, -1.0]
                    ap_pred[frame_nr, :] = [-1.0, -1.0]
                if np.sum(y_true[img_nr, frame_nr, :, :, 1]) > 0:   
                    pp_ref[frame_nr, :] = img_moment(y_true[img_nr, frame_nr, :, :, 1])
                    pp_pred[frame_nr, :] = img_moment(y_pred[img_nr, frame_nr, :, :, 1])
                else:
                    pp_ref[frame_nr, :] = [-1.0, -1.0]
                    pp_pred[frame_nr, :] = [-1.0, -1.0]

                # Transform keypoints to shape [4,1]
                ref = np.concatenate((ap_ref, pp_ref), axis=1)
                pred = np.concatenate((ap_pred, pp_pred), axis=1)
             
            # Append estimated keypoints to list
            keypoints_true.append(ref)
            keypoints_pred.append(pred)
    
    # GlottisNetV2 Channels
    if y_pred.shape[3]>2 and np.ndim(y_pred) == 4:      
        
        for img_nr in range(y_pred.shape[0]):
            ap_ref = np.zeros((int(y_true.shape[-1]/2), 2))
            pp_ref = np.zeros((int(y_true.shape[-1]/2), 2))
            ap_pred = np.zeros((int(y_true.shape[-1]/2), 2))
            pp_pred = np.zeros((int(y_true.shape[-1]/2), 2))
            
            for frame_nr in range(int(y_true.shape[-1]/2)):
                
                # If reference image of prediction map of anterior point is not empty, calculate image moment
                # else write -1 to keypoint list to recognize it later
                if np.sum(y_true[img_nr, :, :, frame_nr]) > 0:
                    ap_ref[frame_nr, :] = img_moment(y_true[img_nr, :, :,frame_nr])
                    ap_pred[frame_nr, :] = img_moment(y_pred[img_nr, :, :, frame_nr])
                else:
                    ap_ref[frame_nr, :] = [-1.0, -1.0]
                    ap_pred[frame_nr, :] = [-1.0, -1.0]

                if np.sum(y_true[img_nr, :, :,  frame_nr+int(y_true.shape[-1]/2)]) > 0: 
                    pp_ref[frame_nr, :] = img_moment(y_true[img_nr, :, :, frame_nr+int(y_true.shape[-1]/2)])
                    pp_pred[frame_nr, :] = img_moment(y_pred[img_nr, :, :, frame_nr+int(y_true.shape[-1]/2)])
                else:
                    pp_ref[frame_nr, :] = [-1.0, -1.0]
                    pp_pred[frame_nr, :] = [-1.0, -1.0]
                
                # Transform keypoints to shape [4,1]
                ref = np.concatenate((ap_ref, pp_ref), axis=1)
                pred = np.concatenate((ap_pred, pp_pred), axis=1)
             
            # Append estimated keypoints to list
            keypoints_true.append(ref)
            keypoints_pred.append(pred)
         
    mapes = MAPE3D(keypoints_true, keypoints_pred, 4)
    mapes = np.asarray(mapes)
    counter=np.count_nonzero(mapes == -1)
    
    if (counter > 0 and len(mapes[mapes != -1])>=1):
        avg = np.mean(mapes[mapes != -1])
        mapes[mapes == -1] = avg
    
    if len(mapes[mapes != -1])==0 and counter>0:
        mapes=[1.0]
        
    mape = np.mean(mapes)
    return mape


'''Metric to evaluate MAPE
   
   Parameters
   ----------
   y_true: Coordinates of original keypoints --> (1,4)
   y_pred: Coordinates of predicted keypoints --> (1,4)
   
   Returns
   -------
   MAPE'''

def metric_mape(y_true, y_pred):
    
    # Initialize lists for original and predicted keypoints
    keypoints_true = []
    keypoints_pred = []
    
    # Iterate through batch
    for img_nr in range(y_pred.shape[0]):
        
        # If reference image of prediction map of anterior point is not empty, calculate image moment
        # else write -1 to keypoint list to recognize it later
        if np.sum(y_true[img_nr, :, :, 0]) > 0:
            ap_ref = img_moment(y_true[img_nr, :, :, 0])
            ap_pred = img_moment(y_pred[img_nr, :, :, 0])
        else:
            ap_ref = (-1.0, -1.0)
            ap_pred = (-1.0, -1.0)
                        
        if np.sum(y_true[img_nr, :, :, 1]) > 0:           
            pp_ref = img_moment(y_true[img_nr, :, :, 1])
            pp_pred = img_moment(y_pred[img_nr, :, :, 1])
        else:
            pp_ref = (-1.0, -1.0)
            pp_pred = (-1.0, -1.0)
        
        # Transform keypoints to shape [4,1]
        ref = np.asarray(ap_ref + pp_ref)
        pred = np.asarray(ap_pred + pp_pred)
        
        # Append estimated keypoints to list
        keypoints_true.append(ref)
        keypoints_pred.append(pred)
         
    mapes = MAPE(keypoints_true, keypoints_pred, 4)
    mapes = np.asarray(mapes)
    counter=np.count_nonzero(mapes == -1)
    
    if (counter > 0 and len(mapes[mapes != -1])>=1):
        avg = np.mean(mapes[mapes != -1])
        mapes[mapes == -1] = avg
    
    if len(mapes[mapes != -1])==0 and counter>0:
        print('MAPE', mapes)
        mapes=[1.0]
        
    mape = np.mean(mapes)
    return mape


'''Metric to evaluate MAPE for anterior point
   
   Parameters
   ----------
   y_true: Coordinates of original keypoints --> (1,2)
   y_pred: Coordinates of predicted keypoints --> (1,2)
   
   Returns
   -------
   MAPE of anterior point'''

def mape_ap(y_true, y_pred):
    
    # Initialize lists for original and predicted keypoints
    keypoints_true = []
    keypoints_pred = []

    if np.ndim(y_pred) == 5:
        fr_choice = int(y_pred.shape[1]/2)
        y_pred = y_pred[:, fr_choice, :, :, :]
        y_true = y_true[:, fr_choice, :, :, :]

    if y_pred.shape[3]>2:
        nr_fr = int(y_pred.shape[3]/2)
        fr_choice = int(nr_fr/2)
        y_pred_new=np.zeros((y_pred.shape[0], y_pred.shape[1], y_pred.shape[2], 2))
        y_true_new=np.zeros((y_true.shape[0], y_true.shape[1], y_true.shape[2], 2))
        y_pred_new[:, :, :, 0] = y_pred[:, :, :, fr_choice]
        y_pred_new[:, :, :, 1] = y_pred[:, :, :, nr_fr + fr_choice]
        y_true_new[:, :, :, 0] = y_true[:, :, :, fr_choice]
        y_true_new[:, :, :, 1] = y_true[:, :, :, nr_fr + fr_choice]
        y_true = y_true_new
        y_pred =y_pred_new
       
    for img_nr in range(y_pred.shape[0]):

         # If reference image of prediction map of anterior point is not empty, calculate image moment
        if np.sum(y_true[img_nr, :, :, 0])>0:
            ap_ref = img_moment(y_true[img_nr, :, :, 0])
            ap_pred = img_moment(y_pred[img_nr, :, :, 0])
        else:
            ap_ref = (-1.0, -1.0)
            ap_pred = (-1.0, -1.0)   
        
        # Transform keypoints to shape [4,1]
        ref = np.asarray(ap_ref)
        pred = np.asarray(ap_pred)
        
        # Append estimated keypoints to list
        keypoints_true.append(ref)
        keypoints_pred.append(pred)
    
    # Calculate MAPE and convert list to array
    mapes = MAPE(keypoints_true, keypoints_pred, 2)    
    mapes = np.asarray(mapes)   
    # If entries with value -1 are inside the array, 
    # calculate average of remaining mapes and set it to position
    counter= np.count_nonzero(mapes == -1)
    
    if (counter > 0 and len(mapes[mapes != -1])>=1):
        avg = np.mean(mapes[mapes != -1])
        mapes[mapes == -1] = avg
    
    if len(mapes[mapes != -1])==0 and counter>0:
        print('AP', mapes)
        mapes=[1.0]
    
    # Calculate average of mapes and return it
    mape = np.mean(mapes)
    return mape


'''Metric to evaluate MAPE for posterior point
   
   Parameters
   ----------
   y_true: Coordinates of original keypoints --> (1,2)
   y_pred: Coordinates of predicted keypoints --> (1,2)
   
   Returns
   -------
   MAPE of posterior point'''

def mape_pp(y_true, y_pred):
    
    # Initialize lists for original and predicted keypoints
    keypoints_true = []
    keypoints_pred = []
    
    if np.ndim(y_pred) == 5:
        fr_choice = int(y_pred.shape[1]/2)
        y_pred = y_pred[:, fr_choice, :, :, :]
        y_true= y_true[:, fr_choice, :, :, :]

    if y_pred.shape[3]>2:
        nr_fr = int(y_pred.shape[3]/2)
        fr_choice = int(nr_fr/2)
        y_pred_new=np.zeros((y_pred.shape[0], y_pred.shape[1], y_pred.shape[2], 2))
        y_true_new=np.zeros((y_true.shape[0], y_true.shape[1], y_true.shape[2], 2))
        y_pred_new[:, :, :, 0] = y_pred[:, :, :, fr_choice]
        y_pred_new[:, :, :, 1] = y_pred[:, :, :, nr_fr + fr_choice]
        y_true_new[:, :, :, 0] = y_true[:, :, :, fr_choice]
        y_true_new[:, :, :, 1] = y_true[:, :, :, nr_fr + fr_choice]
        y_true = y_true_new
        y_pred =y_pred_new
            
    # Iterate through keypoints
    for img_nr in range(y_pred.shape[0]):
        
        # Check if original image is empty. 
        # If not calculate image moment.
        if np.sum(y_true[img_nr, :, :, 1])>0:       
            pp_ref = img_moment(y_true[img_nr, :, :, 1])
            pp_pred = img_moment(y_pred[img_nr, :, :, 1])
        
        # Else set keypoints to -1 to recognize them later
        else:
            pp_ref = (-1.0, -1.0)
            pp_pred = (-1.0, -1.0)       
        
        # Transform keypoints to shape [4,1]
        ref = np.asarray(pp_ref)
        pred = np.asarray(pp_pred)
        
        # Append estimated keypoints to list
        keypoints_true.append(ref)
        keypoints_pred.append(pred)
    
    # Calculate mape of predicted keypoints, covert list to array and count -1 entries       
    mapes = MAPE(keypoints_true, keypoints_pred, 2)    
    mapes = np.asarray(mapes)   
    counter = np.count_nonzero(mapes == -1)
    
    # If entries with value -1 are inside the array, 
    # calculate average of remaining mapes and set it to position
    if (counter > 0 and len(mapes[mapes != -1])>=1):
        avg = np.mean(mapes[mapes != -1])
        mapes[mapes == -1] = avg
    
    if len(mapes[mapes != -1])==0 and counter>0:
        print('PP', mapes)
        mapes=[1.0] 
    
    # Calculate average of mapes and return it
    mape = np.mean(mapes)
    return mape


'''
Used in TrainingChannles.ipynb, Training3DConv.ipynb and Training_LSTM.ipynb.
Load anterior and posterior points to dictionary. 

Parameters
----------
aplist: JSON-file
    Contains anterior and posterior points of each image.
nr: video id

Returns
-------
Dictionary with three columns: video id, coordinates of anterior and coordinates of posterior points.
'''

def load_video_ap(aplist, nr):
    
    # Load jason files
    with open(aplist) as ap_file:
        df_ap = json.load(ap_file)

    # Convert data from json file to pd.dataframe
    df_ap = pd.DataFrame(df_ap['rois'])

    # Create dictionary containing the video id and the coordinates of anterior and posterior points.
    new_row = {'z': nr, 'ap': df_ap.iloc[1]['pos'], 'pp': df_ap.iloc[0]['pos']}

    return new_row


'''
Used in Training2DComp.ipynb
Load anterior and posterior points to dictionary. 

Parameters
----------
aplist: JSON-file
    Contains anterior and posterior points of each image.
nr: video id

Returns
-------
Dictionary with three columns: video id, coordinates of anterior and coordinates of posterior points.
'''

def load_video_2D(aplist, nr, id_nr):
   
    # Load jason files
    with open(aplist) as ap_file:
        df_ap = json.load(ap_file)

    # Convert data from json file to pd.dataframe
    df_ap = pd.DataFrame(df_ap['rois'])

    new_row = {'z': nr, 'id': id_nr, 'ap': df_ap.iloc[1]['pos'], 'pp': df_ap.iloc[0]['pos']}

    return new_row


'''Metric to evaluate MAPE with coordinates as input (GlottisNetV1)
   
   Parameters
   ----------
   y_true: Coordinates of original keypoints --> (1,4)
   y_pred: Coordinates of predicted keypoints --> (1,4)
   
   Returns
   -------
   MAPE'''

def MAPE_V1(keypoints_orig, keypoints_pred):
    
    keys_orig = np.asarray(keypoints_orig)
    keys_pred = np.asarray(keypoints_pred)
    mapes = MAPE(keys_orig, keys_pred, 4)
    mapes = np.asarray(mapes)
    counter=np.count_nonzero(mapes == -1)
    
    if (counter > 0 and len(mapes[mapes != -1])>=1):
        avg = np.mean(mapes[mapes != -1])
        mapes[mapes == -1] = avg
    
    if len(mapes[mapes != -1])==0 and counter>0:
        mapes=[1.0]
        
    mape = np.mean(mapes)
    return mape


'''Metric to evaluate MAPE for anterior point with coordinates as input (GlottisNetV1)
   
   Parameters
   ----------
   y_true: Coordinates of original keypoints --> (1,2)
   y_pred: Coordinates of predicted keypoints --> (1,2)
   
   Returns
   -------
   MAPE for anterior point'''

def mape_apV1(y_true, y_pred):
    keys_true = np.asarray(y_true[:, 0:2])
    keys_pred = np.asarray(y_pred[:, 0:2])
    mapes = MAPE(keys_true, keys_pred, 2)    
    mapes = np.asarray(mapes)   
    
    # If entries with value -1 are inside the array, 
    # calculate average of remaining mapes and set it to position
    counter= np.count_nonzero(mapes == -1)
    
    if (counter > 0 and len(mapes[mapes != -1])>=1):
        avg = np.mean(mapes[mapes != -1])
        mapes[mapes == -1] = avg
    
    if len(mapes[mapes != -1])==0 and counter>0:
        mapes=[1.0]
    
    # Calculate average of mapes and return it
    mape = np.mean(mapes)
    return mape


'''Metric to evaluate MAPE for posterior point with coordinates as input (GlottisNetV1)
   
   Parameters
   ----------
   y_true: Coordinates of original keypoints --> (1,2)
   y_pred: Coordinates of predicted keypoints --> (1,2)
   
   Returns
   -------
   MAPE for posterior point'''

def mape_ppV1(y_true, y_pred):
    keys_true = np.asarray(y_true[:, 2:4])
    keys_pred = np.asarray(y_pred[:, 2:4])
    mapes = MAPE(keys_true, keys_pred, 2)    
    mapes = np.asarray(mapes)   
    # If entries with value -1 are inside the array, 
    # calculate average of remaining mapes and set it to position
    counter= np.count_nonzero(mapes == -1)
    
    if (counter > 0 and len(mapes[mapes != -1])>=1):
        avg = np.mean(mapes[mapes != -1])
        mapes[mapes == -1] = avg
    
    if len(mapes[mapes != -1])==0 and counter>0:
        mapes=[1.0]
    
    # Calculate average of mapes and return it
    mape = np.mean(mapes)
    return mape

