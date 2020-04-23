import os
import numpy as np
import pandas as pd

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

ds = 5
look_back = 10
time_to_dis = np.linspace(0,0.004,100)
no_of_dis = np.array([])
warn = int(250/ds)
datapath = 'C:/Users/Zicheng Liu/Documents/GitHub/hbtepML/Test data/'
predictions = pd.read_pickle('predictions.pkl')
shotno = np.array([])
columns = ["t", "P"]
threshold = 0.73
for filename in os.listdir(datapath):
    shotno = np.append(shotno, int(filename[:-4]))
    


for ttd in time_to_dis: 
    counter = 0
    for sn in shotno: 
        predictions_df_shotno = predictions.loc[predictions['shotno']==sn][columns]
        predictions_df_shotno = predictions_df_shotno.to_numpy().transpose()
        t = predictions_df_shotno[0]
        P = predictions_df_shotno[1]
        ttd_idx = find_nearest(t,t[len(t)-1]-ttd)
        t = t[0:ttd_idx+1]
        P = P[0:ttd_idx+1]
        threshold_np = np.full((len(t),), threshold)
        idx = np.argwhere(np.diff(np.sign(threshold_np - P))).flatten()
        if len(idx) == 0:
            continue
        elif len(idx) >= 1:
            counter += 1
    no_of_dis = np.append(no_of_dis,counter)
#%%
import matplotlib.pyplot as plt
plt.figure()
plt.plot(time_to_dis,no_of_dis/392, c='k')
plt.title("Fraction of detected disruptive shots (f) vs. Time to disruption (t)")
plt.xlabel("t")
plt.ylabel("f")
plt.ylim(0,1)
plt.grid()
plt.show()