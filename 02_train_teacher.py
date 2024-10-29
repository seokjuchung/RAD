import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, models
import qkeras
from qkeras import *
from models import TeacherAutoencoder
import gc


# Open the HDF5 file
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
    teacher = TeacherAutoencoder((nwire//v_split, ntimeticks//f_downsample//h_split, 1)).get_model()
    teacher.compile(optimizer = keras.optimizers.Adam(learning_rate=0.001), loss = 'mse')


filesizes = [17,20,25,23,18,19,21,24,22,24,23,20,23,20,21,23,21,25]

for filenum in range(1):  # 0 to 17
    print(f'file:{filenum}')
    
    # for batchNum in range(filesizes[filenum]):
    for batchNum in range(1):

        file_path = f'./inputData/bnb_WithWire_{filenum:02d}_batch_{batchNum:02d}_{h_split}X{v_split}.npy'

        print(f'run:{batchNum}')
        
        X = np.load(file_path)

        X_train_val, X_test = train_test_split(X,test_size=test_ratio)
        X_train, X_val = train_test_split(X_train_val,test_size=val_ratio / (val_ratio + train_ratio))

        history = teacher.fit(X_train, X_train,
                        epochs = int(h_split*v_split*train_ratio),
                        validation_data = (X_val, X_val),
                        batch_size = int(nbatch))


        del X, X_train, X_val, X_train_val
        tf.keras.backend.clear_session()
        gc.collect() #garbage collector collect
            
teacher.save(f'./savedModel/teacher_{h_split}X{v_split}')
print("saved")