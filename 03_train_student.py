# Train Teacher NN
# 24-01-17: Input size 18X16
# 24-03-01: Use for 864X64 (10X4) split

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, models
from qkeras import *
from models import Student
import gc
from utils import loss


ntimeticks = 6400
nwire = 3456
f_downsample = 10
h_split = 40
v_split = 192
nbatch = 128

train_ratio = 0.5
val_ratio = 0.1
test_ratio = 1 - train_ratio - val_ratio


# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

# Open a strategy scope.
with strategy.scope():
    # Everything that creates variables should be under the strategy scope.
    # In general this is only model construction & `compile()`.
    teacher = tf.keras.models.load_model(f'./savedModel/teacher_{h_split}X{v_split}')
    student = Student((nwire//v_split*ntimeticks//f_downsample//h_split,)).get_model()
    student.compile(optimizer = 'adam', loss = 'mse')

filesizes = [17,20,25,23,18,19,21,24,22,24,23,20,23,20,21,23,21,25]

for filenum in range(1):  # 0 to 17

    # for batchNum in range(filesizes[filenum]):
    for batchNum in range(1):
        file_path = f'./inputData/bnb_WithWire_{filenum:02d}_batch_{batchNum:02d}_{h_split}X{v_split}.npy'

        print(f'run:{batchNum}')
        
        X = np.load(file_path)

        X_train_val, X_test = train_test_split(X,test_size=test_ratio)
        X_train, X_val = train_test_split(X_train_val,test_size=val_ratio / (val_ratio + train_ratio))

        X_train_predict_teacher = teacher.predict(X_train)
        X_val_predict_teacher = teacher.predict(X_val)
        X_train_loss_teacher = loss(X_train, X_train_predict_teacher, 'mse')
        X_val_loss_teacher = loss(X_val, X_val_predict_teacher, 'mse')

        history = student.fit(X_train.reshape((-1,nwire*ntimeticks//f_downsample//v_split//h_split,1)), X_train_loss_teacher,
        epochs = int(h_split*v_split*train_ratio),
        validation_data = (X_val.reshape((-1,nwire*ntimeticks//f_downsample//v_split//h_split,1)), X_val_loss_teacher),
        batch_size = int(nbatch))

        del X, X_train, X_val, X_train_val
        tf.keras.backend.clear_session()
        gc.collect() #garbage collector collect
            
student.save(f'./savedModel/student_{h_split}X{v_split}')
print("saved")