from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow as tf
import tensorflow_addons as tfa

class ModelSaver(Callback):
    def __init__(self, model, N, steps_path):
        self.model=model
        self.N=N
        self.epoch=0
        self.steps_path = steps_path
        
    def on_epoch_end(self, epoch, logs={}):
        if self.epoch % self.N==0:
            name=self.steps_path + "/steps/epoch%03d.h5" %self.epoch
            self.model.save(name)
        self.epoch+=1
        
# Used learning rate scheduler
def scheduler(epoch):   
    if epoch <5:
        return 1e-3
    elif epoch <15:
        return 2e-4
    elif epoch <25:
        return 1e-4
    else:
        return 5e-5

def get_callbacks(MODEL_PATH, model, N, steps_path):
    
    """
    Saves model if MAPE decreases.
    """
    csv_logger = CSVLogger(MODEL_PATH.replace(".h5", ".csv"))
    checkpoint = ModelCheckpoint(MODEL_PATH, 
                                 monitor = 'val_ap_pred_metric_mape3D',
                                 verbose = 1, 
                                 save_best_only = True, 
                                 mode='min')
    save_model = ModelSaver(model, N, steps_path)    
    tb_callback = tf.keras.callbacks.TensorBoard(steps_path + './logs', update_freq=1)
    callbacks = [checkpoint, csv_logger, save_model, tb_callback]

    return callbacks
