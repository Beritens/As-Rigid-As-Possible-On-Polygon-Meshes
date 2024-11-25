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

#36
data36GD = pd.read_csv('./36GD.csv', header=None, delimiter=",")
data36GD.columns = ['Index', 'Time_ns', 'DistantTime', 'DescentTime', 'Energy']
#68
data68GD = pd.read_csv('./68GD.csv', header=None, delimiter=",")
data68GD.columns = ['Index', 'Time_ns', 'DistantTime', 'DescentTime', 'Energy']
#100
data100GD = pd.read_csv('./100GD.csv', header=None, delimiter=",")
data100GD.columns = ['Index', 'Time_ns', 'DistantTime', 'DescentTime', 'Energy']
#164
data164GD = pd.read_csv('./164GD.csv', header=None, delimiter=",")
data164GD.columns = ['Index', 'Time_ns', 'DistantTime', 'DescentTime', 'Energy']
#228
data228GD = pd.read_csv('./228GD.csv', header=None, delimiter=",")
data228GD.columns = ['Index', 'Time_ns', 'DistantTime', 'DescentTime', 'Energy']
#356
data356GD = pd.read_csv('./356GD.csv', header=None, delimiter=",")
data356GD.columns = ['Index', 'Time_ns', 'DistantTime', 'DescentTime', 'Energy']
#484
data484GD = pd.read_csv('./484GD.csv', header=None, delimiter=",")
data484GD.columns = ['Index', 'Time_ns', 'DistantTime', 'DescentTime', 'Energy']
#644
data644GD = pd.read_csv('./644GD.csv', header=None, delimiter=",")
data644GD.columns = ['Index', 'Time_ns', 'DistantTime', 'DescentTime', 'Energy']

conjugateGradientDescentTimeMeans = [data36["DescentTime"].mean(), data68["DescentTime"].mean(), data100["DescentTime"].mean(),
                 data164["DescentTime"].mean(), data228["DescentTime"].mean(), data356["DescentTime"].mean(),
                 data484["DescentTime"].mean(), data644["DescentTime"].mean()];

gradientDescentTimeMeans = [data36GD["DescentTime"].mean(), data68GD["DescentTime"].mean(), data100GD["DescentTime"].mean(),
                 data164GD["DescentTime"].mean(), data228GD["DescentTime"].mean(), data356GD["DescentTime"].mean(),
                 data484GD["DescentTime"].mean(), data644GD["DescentTime"].mean()];
sizes = [36,68,100,164,228,356,484,644];

plt.figure(figsize=(10, 5))
plt.plot(sizes, gradientDescentTimeMeans, linestyle='-', color='b', marker='o', label='Gradient Descent')
plt.plot(sizes, conjugateGradientDescentTimeMeans, linestyle='-', color='r', marker='o', label='Newton-like method')
plt.xlabel('Mesh Vertices')
plt.ylabel('Time (ns)')
plt.title('Gradient Descent vs Newton-like method')
plt.yscale("log")
plt.legend()
plt.grid(False)
plt.show()
