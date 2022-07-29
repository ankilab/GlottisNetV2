

Preparation of training data
****************************
GlottisNetV1 and GlottisNetV2
-----------------------------

Necessary data for training:
    * Images of the glottis 
    * Segmentation of glottal area
    * Annotations of anterior and posterior points

GlottisNetV1 and GlottisNetV2 are trained on the openly available BAGLS data set that can be found in https://github.com/anki-xyz/bagls. The annotations of the 
anterior and posterior points are provided in https://github.com/anki-xyz/GlottalMidline. The coordinates need to be stored as JSON.

GlottisNetV2 trained on videos
------------------------------

The used videos can be found in .... .

The videos are stored as mp4-files and have 30 frames each. For the annotation of the anterior and posterior points, the 
in https://github.com/anki-xyz/GlottalMidline provided tool, was used. The segmentations
were created using a trained model of GlottisNetV2 on the frames of the input videos. The corresponding Python script *predict_segemntation.py* is available 
in this repository in the directory "Examples". Those precalculated segmentations were enhanced and revised sing PiPra, provided in https://github.com/anki-xyz/pipra.