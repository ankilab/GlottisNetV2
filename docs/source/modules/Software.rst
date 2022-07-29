

Software Setup
**************

Requirements
------------
* Windows 10
* Python 3.x (ideally Python 3.8)
* Anaconda package


Installation
------------

Clone the GitHub repository, create anaconda environment including Python and install the repository::

    pip install git+https://github.com/ankilab/GlottisNetV2.git

Due to version conflicts inside the pip install environment, please install the following packages according to your system/needs:

* TensorFlow (with or without GPU support) in v.2.5+
* `segmentation_models <https://github.com/qubvel/segmentation_models>`_ `pip install segmentation-models`
* `segmentation_models_3D <https://github.com/ZFTurbo/segmentation_models_3D>`_ `pip install segmentation-models-3D`



Training GlottisNetV1 and GlottisNetV2
--------------------------------------
The used notebooks for training are provided in the "Examples" directory.

Used notebooks:
    * Training_GlottisNetV1.ipynb
    * Training_GlottisNetV2.ipynb


Set path to training data and to desired location of the final model inside the notebook. Define the parameters and execute the notebook. 
When training *GlottisNetV2c* the output of the DataGenerator needs to be adapted. The code is provided in *Utils/DataGenerator.py*. Remove the comment 
and restart the notebook.


Training GlottisNetV2 on videos
-------------------------------
Used notebooks:
    * Training_Channels.ipynb (Training of GlottisNetV2 Channels)
    * Training_3DConv.ipynb (Trainining of GlottisNetV2 3DConv)
    * Training_LSTM.ipynb (Training of GlottisNetV2 LSTM)
    * Training_3DReference.ipynb (Training of final 2D-version of GlottisNetV2 on videos)
    * Training_3DGlottisNetV1.ipynb (Training of GlottisNetV1 on videos)

Define desired parameters and path of training and model location. When the number of frames is changed (except Training_3DReference.ipynb and TrainingGlottisNetV1.ipynb), 
additional changes in the corresponding DataGenerators need to be made
for the correct implementation of the augmentation. The additional frames have to be added as additional targets. This has to be done manually in *Utils/Datagenerator_XXX.py*.





