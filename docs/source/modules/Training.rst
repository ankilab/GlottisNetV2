Helper functions for Training of GlottisNetV2
********************************************

Some helper functions for the training are contained in Utils/data.py

* load_data(aplist, n):
    Load anterior and posterior points to Dataframe. 
        * Parameters:
            * aplist (JSON-file): Contains anterior and posterior points of each image (output of annotation tool)
            * n (int): number of training images
        * Returns:
            * DataFrame with three columns: image id, coordinates of anterior, and coordinates of posterior points.

* img_moment(img):
    Calculate mass of input image
        * Parameters:
            * img: Prediction map of anterior and posterior point
        * Returns:
            * Coordinates of mass of input image

* MAPE(keypoints_orig, keypoints_pred, nr_points, eps = 1e-9):
    Calculate MAPE
        * Parameters:
            * keypoints_orig: Coordinates of original keypoints
            * keypoints_pred: Coordinates of predicted keypoints
            * nr_points: number of coordinates (in this case 4, two for anterior and posterior points each)
        * Returns:
            * MAPE

* MAPE3D(keypoints_orig, keypoints_pred, nr_points, eps = 1e-9):
    Calculate MAPE of several frames
        * Parameters:
            * keypoints_orig: Coordinates of original keypoints --> 5-dimensional array or 4-dimensional array with more than 3 channels
            * keypoints_pred: Coordinates of predicted keypoints --> 5-dimensional array or 4-dimensional array with more than 3 channels
            * nr_points: number of coordinates (in this case 4, two for anterior and posterior points each)
        * Returns:
            * MAPE

* metric_mape3D(y_true, y_pred):
    Metric to evaluate MAPE of several frames (custom metric). The input images have 5 dimensions or 4 dimensions and more than 2 channels.
        * Parameters
            * y_true: Coordinates of original keypoints --> (1,4)
            * y_pred: Coordinates of predicted keypoints --> (1,4)
        * Returns: 
            * MAPE

 * metric_mape(y_true, y_pred):
    Metric to evaluate MAPE (custom metric)  
        * Parameters: 
            * y_true: Coordinates of original keypoints --> (1,4)
            * y_pred: Coordinates of predicted keypoints --> (1,4)
   
        * Returns:
            * MAPE

* mape_ap(y_true, y_pred):
    Metric to evaluate MAPE of anterior point (custom metric)
        * Parameters: 
            * y_true: Coordinates of original keypoints --> (1,2)
            * y_pred: Coordinates of predicted keypoints --> (1,2)
   
        * Returns:
            * MAPE of anterior point

* mape_ap(y_true, y_pred):
    Metric to evaluate MAPE of posterior point (custom meric)
        * Parameters: 
            * y_true: Coordinates of original keypoints --> (1,2)
            * y_pred: Coordinates of predicted keypoints --> (1,2)
   
        * Returns:
            * MAPE of posterior point

* load_video_ap(aplist, nr):
    Used in TrainingChannles.ipynb, Training3DConv.ipynb and Training_LSTM.ipynb. Load anterior and posterior points to dictionary.
        * Parameters:
            * aplist: JSON-file
                * Contains anterior and posterior points of each image.
            * nr: int
                * video id
        * Returns:
            * Dictionary with three columns: video id, coordinates of anterior, and coordinates of posterior points.

* load_video_2D(aplist, nr, id_nr):
    Used in Training2DComp.ipynb and Training_3DGlottisNetV1. Load anterior and posterior points to dictionary. 
        * Parameters
            * aplist: JSON-file
                * Contains anterior and posterior points of each image.
            * id_nr: int
                * video id
        * Returns: 
            * Dictionary with three columns: video id, coordinates of anterior and coordinates of posterior points.

* MAPE_V1(keypoints_orig, keypoints_pred):
    Metric to evaluate MAPE with coordinates as input (GlottisNetV1)
        * Parameters:
            * y_true: Coordinates of original keypoints --> (1,4)
            * y_pred: Coordinates of predicted keypoints --> (1,4)
        * Returns:
            * MAPE

* MAPE_apV1(keypoints_orig, keypoints_pred):
    Metric to evaluate MAPE with coordinates as input (GlottisNetV1) for anterior point
        * Parameters:
            * y_true: Coordinates of original keypoints --> (1,2)
            * y_pred: Coordinates of predicted keypoints --> (1,2)
        * Returns:
            * MAPE for anterior point

* MAPE_ppV1(keypoints_orig, keypoints_pred):
    Metric to evaluate MAPE with coordinates as input (GlottisNetV1) for posterior point
        * Parameters:
            * y_true: Coordinates of original keypoints --> (1,2)
            * y_pred: Coordinates of predicted keypoints --> (1,2)
        * Returns:
            * MAPE for posterior point



    

