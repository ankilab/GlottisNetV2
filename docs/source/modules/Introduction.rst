Introduction
************

GlottisNetV2 is a multi-task architecture to retrieve two of the most important features from laryngeal endoscopy videos, the 
glottal midline and the segmentation of the glottal area. The architecture of GlottisNetV2 is based on a U-Net and uses prediction maps for visualizing the probability that anterior and posterior points are located in that specific area of the image. It has two separate decoders for segmenting the glottal area and predicting the anterior and posterior points.

This user guide contains explanations of the following topics:
    * Models
    * Software setup
    * Training GlottisNetV2