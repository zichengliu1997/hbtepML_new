from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch
import numpy as np
import os

SEED = 1
MAX_TRIALS = 20
EXECUTION_PER_TRIAL = 2
N_EPOCH_SEARCH = 100
numCLasses = 2

look_back = 10
batch_size = 1000
early_pat = 6
reduced_pat = 3
model_name = "LSTM_model.h5"
cd = os.getcwd()

tdata_np_3d = np.load('train_data.npy')
train_label = np.load('label.npy')
vdata_np_3d = np.load('val_data.npy')
validate_label = np.load('val_label.npy')
no_of_param = np.shape(tdata_np_3d)[2]
epochs = 100
class LSTMModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = models.Sequential()
        model.add(layers.LSTM(units=hp.Int('units1',min_value=16,max_value=512,step=16),activation='relu', input_shape=(look_back, no_of_param), return_sequences=True))
        model.add(layers.Dropout(rate=hp.Float('dropout_1',min_value=0.0,max_value=0.5,default=0.2,step=0.05,)))
        model.add(layers.LSTM(units=hp.Int('units2',min_value=16,max_value=512,step=16), activation='relu', return_sequences=True))
        model.add(layers.Dropout(rate=hp.Float('dropout_2',min_value=0.0,max_value=0.5,default=0.2,step=0.05,)))
        model.add(layers.LSTM(2, activation='softmax', return_sequences=False))

        model.compile(
            optimizer=optimizers.Adam(
                hp.Float(
                    'learning_rate',
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling='LOG',
                    default=1e-3
                )
            ),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

callbacks_list = [
    EarlyStopping(monitor = 'val_accuracy', patience = early_pat, restore_best_weights = True), 
    ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.1, patience = reduced_pat)]
lstmmodel = LSTMModel(input_shape=(look_back, no_of_param), num_classes=numCLasses)
tuner = RandomSearch(lstmmodel,objective='val_accuracy',seed=SEED,max_trials=MAX_TRIALS,executions_per_trial=EXECUTION_PER_TRIAL,directory='random_search',project_name='HBTEPDisruption')
EXECUTION_PER_TRIAL = 2
tuner.search_space_summary()
tuner.search(tdata_np_3d, train_label, epochs=N_EPOCH_SEARCH, validation_data = (vdata_np_3d, validate_label),batch_size = batch_size, callbacks =callbacks_list)
tuner.results_summary()
best_model = tuner.get_best_models(num_models=1)[0]

edata_np_3d = np.load('eval_data.npy')
eval_label = np.load('eval_label.npy')
loss, accuracy = best_model.evaluate(edata_np_3d, eval_label)
best_model.save(model_name)