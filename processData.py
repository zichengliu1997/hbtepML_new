from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Define some parameters
ds = 5
look_back = 10
utp = int(1000/ds)
t_warn = 250/ds
problematic = [100127,100655,100663,100991,105859]
columns = ["ip","q","mr","lv","spect","sxrmid","n1amp","bpli","oh","vf"]

# Temporalize into 3D
def temporalizeShot(data_df_slice, lookback):
    data_np_slice_3d = np.empty((0, lookback, data_df_slice.shape[1]))
    
    for i in np.arange(lookback, data_df_slice.shape[0]):
        data_np_slice_3d = np.append(data_np_slice_3d, [data_df_slice.iloc[i:i-lookback:-1,:].to_numpy()], axis=0)
    return data_np_slice_3d

# Load the data file
data_file_name = 'hbtepdata_nn_data.csv'
data = pd.read_csv(data_file_name)
#%%
# Getting rid of problematic shots
for i in problematic:
    data = data[data.shotno != i]
data = data.reset_index()
# Saving the shotno column and t column into separate dataframes for later use
shotno_repeated = data[["shotno"]]
shottime = data[["t"]]
#%%
# Split shot numbers into training, validation, evaluation, and testing sets
shotno_list = np.unique(data['shotno'].to_numpy())
nshot = len(shotno_list)
train_shotno, test_shotno= train_test_split(shotno_list, test_size=0.1, random_state=42)
val_percent = len(test_shotno)/len(train_shotno)
train_shotno, val_shotno = train_test_split(train_shotno, test_size=val_percent, random_state=42)
eval_percent = len(val_shotno)/len(train_shotno)
train_shotno, eval_shotno = train_test_split(train_shotno, test_size=eval_percent, random_state=42)
no_of_param = len(columns)
#%%
# Empty training, validation, and evaluation data sets
tdata_np_3d = np.empty((0,look_back, no_of_param))
vdata_np_3d = np.empty((0,look_back, no_of_param))
edata_np_3d = np.empty((0,look_back, no_of_param))

# Generating training dataframe
traindf = pd.DataFrame()
tcounter = 0
tlength = len(train_shotno)
for shotno in train_shotno:
    traindf = traindf.append(data.loc[data['shotno']==shotno][columns])
    tcounter += 1
    print(str(tcounter) + '/' + str(tlength))

# Normalizing and saving a file of traindf
scaler = preprocessing.StandardScaler()
params = scaler.fit(traindf)
traindf.to_pickle("traindf.pkl")
traindf= pd.DataFrame(scaler.transform(traindf))

# Getting rid of shotno and t columns from data temporarily and normalizing it
data = data[columns]
data = pd.DataFrame(scaler.transform(data))

# Adding back the shotno and t columns
data["shotno"] = shotno_repeated
data["t"] = shottime

# Preparing the training, validation, evaluation, and testing data
tcounter = 0
vcounter = 0
ecounter = 0
tecounter = 0

vlength = len(val_shotno)
elength = len(eval_shotno)
telength = len(test_shotno)
cd = os.getcwd()
data.columns = ["ip","q","mr","lv","spect","sxrmid","n1amp","bpli","oh","vf", "shotno","t"]

for shotno in train_shotno:
    data_df_shotno = data.loc[data['shotno']==shotno][columns]
    no_t = data_df_shotno.shape[0]
    data_df_shotno = data_df_shotno.iloc[no_t-utp-look_back:no_t]
    data_np_shotno = temporalizeShot(data_df_shotno, look_back)
    tdata_np_3d = np.concatenate((tdata_np_3d, data_np_shotno),axis=0)
    tcounter += 1
    print(str(tcounter) + '/' + str(tlength))

for shotno in val_shotno:
    data_df_shotno = data.loc[data['shotno']==shotno][columns]
    no_t = data_df_shotno.shape[0]
    data_df_shotno = data_df_shotno.iloc[no_t-utp-look_back:no_t]
    data_np_shotno = temporalizeShot(data_df_shotno, look_back)
    vdata_np_3d = np.concatenate((vdata_np_3d, data_np_shotno),axis=0)
    vcounter += 1
    print(str(vcounter) + '/' + str(vlength))
    
for shotno in eval_shotno:
    data_df_shotno = data.loc[data['shotno']==shotno][columns]
    no_t = data_df_shotno.shape[0]
    data_df_shotno = data_df_shotno.iloc[no_t-utp-look_back:no_t]
    data_np_shotno = temporalizeShot(data_df_shotno, look_back)
    edata_np_3d = np.concatenate((edata_np_3d, data_np_shotno),axis=0)
    ecounter += 1
    print(str(ecounter) + '/' + str(elength))

for shotno in test_shotno:
    data_df_shotno = data.loc[data['shotno']==shotno][columns]
    data_np_shotno = temporalizeShot(data_df_shotno, look_back)
    time = data.loc[data['shotno']==shotno]["t"].to_numpy()
    np.save(cd + "\\Test data\\" + str(int(shotno)), data_np_shotno)
    np.save(cd + "\\Test data time\\time_" + str(int(shotno)), time)
    tecounter += 1
    print(str(tecounter) + '/' + str(telength))

# Preparing labels
#%%
tl = np.array(())

for j in range(utp):
    if j < int(utp - t_warn): 
        tl = np.concatenate((tl,np.array([0])))
    else: 
        tl = np.concatenate((tl,np.array([1])))
        
train_label = np.array([])
for i in range(len(train_shotno)):
    train_label = np.concatenate((train_label, tl))

validate_label = np.array([])
for i in range(len(val_shotno)):
    validate_label = np.concatenate((validate_label, tl))
    
evaluate_label = np.array([])
for i in range(len(eval_shotno)):
    evaluate_label = np.concatenate((evaluate_label, tl))
            

train_label = to_categorical(train_label)
validate_label = to_categorical(validate_label)
evaluate_label = to_categorical(evaluate_label)
#%%
# Saving the data and label for NN
np.save('train_data', tdata_np_3d)
np.save('val_data', vdata_np_3d)
np.save('eval_data', edata_np_3d)
np.save('label', train_label)
np.save('val_label', validate_label)
np.save('eval_label', evaluate_label)