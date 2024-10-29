# 24-01-05: Get anomaly score of student and teacher
# 24-01-17: Modified size to 18X16 to get model dimension similar to Princeton
# 24-03-01: Use for 864X64 (10X4) split

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.metrics import roc_curve, auc
import qkeras
from qkeras import *
import gc
from utils import loss

# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

ntimeticks = 6400
nwire = 3456
f_downsample = 10
h_split = 10
v_split = 4
nbatch = 32

adccutoff = 10.*f_downsample/10.
adcsaturation = 100.*f_downsample/10.

# Open a strategy scope.
with strategy.scope():
    # Everything that creates variables should be under the strategy scope.
    # In general this is only model construction & `compile()`.
    teacher = tf.keras.models.load_model(f'./savedModel/teacher_{h_split}X{v_split}')
    student = qkeras.utils.load_qmodel(f'./savedModel/student_{h_split}X{v_split}')

filesizes = [17,20,25,23,18,19,21,24,22,24,23,20,23,20,21,23,21,25]

for filenum in range(1):  # 0 to 17
    loss_teacher = np.array([]) 
    loss_student = np.array([]) 

    for batchNum in range(filesizes[filenum]):
    # for batchNum in range(1):

        file_path = f'/nevis/westside/data/sc5303/Data/{h_split}X{v_split}_npy/bnb_WithWire_{filenum:02d}_pureNu_batch_{batchNum:02d}_{h_split}X{v_split}.npy'

        print(f'run:{batchNum}')
        
        X = np.load(file_path)

        X_predict_teacher = teacher.predict(X)
        X_loss_teacher = loss(X, X_predict_teacher, 'mse')
        X_loss_student = student.predict(X.reshape((-1,nwire*ntimeticks//f_downsample//v_split//h_split,1))).reshape(len(X_loss_teacher))

        loss_teacher = np.append(loss_teacher, X_loss_teacher)
        loss_student = np.append(loss_student, X_loss_student)

        del X, X_predict_teacher, X_loss_teacher, X_loss_student
        tf.keras.backend.clear_session()
        gc.collect() #garbage collector collect
            
        #     break
        # break

    np.save(f'output/reconstruct/4X10/teacher_loss_{filenum}.npy', loss_teacher)
    np.save(f'output/reconstruct/4X10/cicadaV2__loss_{filenum}.npy', loss_student)
