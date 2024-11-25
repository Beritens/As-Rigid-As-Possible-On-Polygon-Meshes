import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

data = pd.read_csv('./GDvCONBigCON.csv', header=None, delimiter=",")
data.columns = ['Index', 'Time_ns', 'DistantTime', 'DescentTime', 'Energy', 'DistStep', 'DescStep']

data2 = pd.read_csv('./GDvCONBigGD.csv', header=None, delimiter=",")
data2.columns = ['Index', 'Time_ns', 'DistantTime', 'DescentTime', 'Energy', 'DistStep', 'DescStep']


plt.figure(figsize=(10, 5))
plt.ylim(0, 16)
plt.xlim(-500000000, 10000000000)
plt.plot(data['Time_ns'], data['Energy'], linestyle='-', color='b', marker='o', label='Newton-like method')
plt.plot(data2['Time_ns'], data2['Energy'], linestyle='-', color='r', marker='o', label='gradient descent')
plt.xlabel('Time (ns)')
plt.ylabel('Energy')
plt.title('Gradient Descent vs Newton-like method - Big Meshes')
plt.legend()
plt.grid(False)
plt.show()

