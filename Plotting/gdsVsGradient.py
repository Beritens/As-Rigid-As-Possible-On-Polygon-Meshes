import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

#36
data36 = pd.read_csv('./36.csv', header=None, delimiter=",")
data36.columns = ['Index', 'Time_ns', 'DistantTime', 'DescentTime', 'Energy']
#68
data68 = pd.read_csv('./68.csv', header=None, delimiter=",")
data68.columns = ['Index', 'Time_ns', 'DistantTime', 'DescentTime', 'Energy']
#100
data100 = pd.read_csv('./100.csv', header=None, delimiter=",")
data100.columns = ['Index', 'Time_ns', 'DistantTime', 'DescentTime', 'Energy']
#164
data164 = pd.read_csv('./164.csv', header=None, delimiter=",")
data164.columns = ['Index', 'Time_ns', 'DistantTime', 'DescentTime', 'Energy']
#228
data228 = pd.read_csv('./228.csv', header=None, delimiter=",")
data228.columns = ['Index', 'Time_ns', 'DistantTime', 'DescentTime', 'Energy']
#356
data356 = pd.read_csv('./356.csv', header=None, delimiter=",")
data356.columns = ['Index', 'Time_ns', 'DistantTime', 'DescentTime', 'Energy']
#484
data484 = pd.read_csv('./484.csv', header=None, delimiter=",")
data484.columns = ['Index', 'Time_ns', 'DistantTime', 'DescentTime', 'Energy']
#644
data644 = pd.read_csv('./644.csv', header=None, delimiter=",")
data644.columns = ['Index', 'Time_ns', 'DistantTime', 'DescentTime', 'Energy']

distTimeMeans = [data36["DistantTime"].mean(), data68["DistantTime"].mean(), data100["DistantTime"].mean(),
                 data164["DistantTime"].mean(), data228["DistantTime"].mean(), data356["DistantTime"].mean(),
                 data484["DistantTime"].mean(), data644["DistantTime"].mean()];

descentTimeMeans = [data36["DescentTime"].mean(), data68["DescentTime"].mean(), data100["DescentTime"].mean(),
                 data164["DescentTime"].mean(), data228["DescentTime"].mean(), data356["DescentTime"].mean(),
                 data484["DescentTime"].mean(), data644["DescentTime"].mean()];
sizes = [36,68,100,164,228,356,484,644];

plt.figure(figsize=(10, 5))
plt.plot(sizes, distTimeMeans, linestyle='-', color='b', marker='o', label='global distance step')
plt.plot(sizes, descentTimeMeans, linestyle='-', color='r', marker='o', label='descent step')
plt.xlabel('Mesh Vertices')
plt.ylabel('Time (ns)')
plt.title('Descent Step vs Global Distance Step')
plt.yscale("log")
plt.legend()
plt.grid(False)
plt.show()
