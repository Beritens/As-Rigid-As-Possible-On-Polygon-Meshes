import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

data = pd.read_csv('./big-con-sar.csv', header=None, delimiter=",")
data.columns = ['Index', 'Time_ns', 'DistantTime', 'DescentTime', 'Energy']

data2 = pd.read_csv('./big-onlyCon-sar.csv', header=None, delimiter=",")
data2.columns = ['Index', 'Time_ns', 'DistantTime', 'DescentTime', 'Energy']

data3 = pd.read_csv('./big-block-sar.csv', header=None, delimiter=",")
data3.columns = ['Index', 'Time_ns', 'DistantTime', 'DescentTime', 'Energy']



plt.figure(figsize=(10, 5))
plt.ylim(0, 20)
plt.xlim(0, data2["Time_ns"].max())
plt.plot(data['Time_ns'], data['Energy'], linestyle='-', color='b', marker='o', label='descent + global distance step')
plt.plot(data2['Time_ns'], data2['Energy'], linestyle='-', color='r', marker='o', label='only descent')
plt.plot(data3['Time_ns'], data3['Energy'], linestyle='-', color='g', marker='o', label='block method')
plt.xlabel('Time (ns)')
plt.ylabel('Energy')
plt.title('Pure Descent vs Global Distance Step vs Block-Descent')
plt.legend()
plt.grid(False)
plt.show()

