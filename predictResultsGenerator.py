from tensorflow.keras.models import load_model
import os
import numpy as np
import pandas as pd

look_back = 10
model = load_model('LSTM_model.h5')
datapath = 'C:/Users/Zicheng Liu/Documents/GitHub/hbtepML/Test data/'
timepath = 'C:/Users/Zicheng Liu/Documents/GitHub/hbtepML/Test data time/'
predictions = pd.DataFrame()
counter = 0
for filename in os.listdir(datapath):
    os.path.join(datapath, filename)
    test_data_np = np.load(os.path.join(datapath, filename))
    time = np.load(os.path.join(timepath, 'time_'+filename))
    results = model.predict(test_data_np)
    results = results.transpose()
    probability = results[1]
    t = time[look_back:]
    shotno=np.full((len(t),), int(filename[:-4]))
    print(filename)
    p_vs_t = pd.DataFrame({'t': t,'P': probability, 'shotno': shotno})
    predictions = predictions.append(p_vs_t)
    counter += 1
    print(str(counter) + " done")
#%%
predictions.to_pickle("predictions.pkl")
#%%
from tensorflow.keras.models import load_model
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

look_back = 10
model = load_model('LSTM_model.h5')
datapath = 'C:/Users/Zicheng Liu/Documents/GitHub/hbtepML/Test data/'
timepath = 'C:/Users/Zicheng Liu/Documents/GitHub/hbtepML/Test data time/'
filename = "100007.npy"
os.path.join(datapath, filename)
test_data_np = np.load(os.path.join(datapath, filename))
time = np.load(os.path.join(timepath, 'time_'+filename))
results = model.predict(test_data_np)
results = results.transpose()
probability = results[1]
t = time[look_back:]
plt.figure()
plt.plot(t*1000,probability)
plt.title("Disruptiion Probability (P(t)) vs. Time from Plasma Breakdown (t)\nShot 100007")
plt.xlabel("t(ms)")
plt.ylabel("P(t)")
plt.grid()
plt.show()