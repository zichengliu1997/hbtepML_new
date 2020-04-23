import os
import numpy as np
import pandas as pd

ds = 5
look_back = 10
threshold_values = np.linspace(0,1,100)
tpno = np.array([])
warn = int(250/ds)
datapath = 'C:/Users/Zicheng Liu/Documents/GitHub/hbtepML/Test data/'
predictions = pd.read_pickle('predictions.pkl')
shotno = np.array([])
columns = ["t", "P"]
for filename in os.listdir(datapath):
    shotno = np.append(shotno, int(filename[:-4]))
tpno = np.array([])
for threshold in threshold_values:
    tp_counter = 0
    for sn in shotno: 
        predictions_df_shotno = predictions.loc[predictions['shotno']==sn][columns]
        predictions_df_shotno = predictions_df_shotno.to_numpy().transpose()
        t = predictions_df_shotno[0]
        P = predictions_df_shotno[1]
        threshold_np = np.full((len(t),),threshold)
        idx = np.argwhere(np.diff(np.sign(threshold - P))).flatten()
        t_warn = t[len(t)-1-warn]
        if len(idx) == 0:
            continue
        elif len(idx) >= 1:
            if t[idx[0]] >= t_warn:
                tp_counter += 1
            else:
                continue
    tpno = np.append(tpno,tp_counter)
#%%
import matplotlib.pyplot as plt
plt.figure()
plt.plot(threshold_values,tpno/392, c='k')
plt.scatter(threshold_values[np.argmax(tpno)], tpno[np.argmax(tpno)]/392, c='r')
plt.title("True positive rate (TPR) vs. Threshold")
plt.xlabel("Threshold (0 - 1)")
plt.ylabel("TPR")
plt.ylim(0,1)
plt.grid()
plt.show()
#%%
print("The best threshold is: " + str(threshold_values[np.argmax(tpno)]))