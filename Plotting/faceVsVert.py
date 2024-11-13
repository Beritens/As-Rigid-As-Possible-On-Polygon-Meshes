import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

data = pd.read_csv('./FaceVsVertFace.csv', header=None, delimiter=",")
data.columns = ['Index', 'Time_ns', 'DistantTime', 'DescentTime', 'Energy', 'DistStep', 'DescStep']

data2 = pd.read_csv('./FaceVsVertVert.csv', header=None, delimiter=",")
data2.columns = ['Index', 'Time_ns', 'DistantTime', 'DescentTime', 'Energy', 'DistStep', 'DescStep']




plt.figure(figsize=(10, 5))
plt.xlim(0, 4000000000)
plt.plot(data['Time_ns'], data['Energy'], linestyle='-', color='b', marker='o', label='Poly-Face-ARAP')
plt.plot(data2['Time_ns'], data2['Energy'], linestyle='-', color='r', marker='o', label='Spokes-And-Virtual-Rims')
plt.xlabel('Time (ns)')
plt.ylabel('Energy')
plt.title('Poly-Face-ARAP vs Spokes-And-Virtual-Rims')
plt.legend()
plt.grid(False)
plt.show()

