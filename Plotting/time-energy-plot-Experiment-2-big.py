import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

data = pd.read_csv('./Experiment2-big-con.csv', header=None, delimiter=",")
data.columns = ['Index', 'Time_ns', 'DistantTime', 'DescentTime', 'Energy']

data2 = pd.read_csv('./Experiment2-big-gd.csv', header=None, delimiter=",")
data2.columns = ['Index', 'Time_ns', 'DistantTime', 'DescentTime', 'Energy']


plt.figure(figsize=(10, 5))
plt.ylim(0, 30)
plt.plot(data['Time_ns'], data['Energy'], linestyle='-', color='b', marker='o')
plt.plot(data2['Time_ns'], data2['Energy'], linestyle='-', color='r', marker='o')
plt.xlabel('Time (ns)')
plt.ylabel('Energy')
plt.title('Gradient Descent vs Conjugate Gradient Descent - Big Meshes')
plt.grid(False)
plt.show()

