import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

data = pd.read_csv('./stepImprovements.csv', header=None, delimiter=",")
data.columns = ['Index', 'Time_ns', 'DistantTime', 'DescentTime', 'Energy', 'DistStep', 'DescStep']



plt.figure(figsize=(10, 5))
plt.plot(data['Time_ns'], -data['DistStep'], linestyle='-', color='b', marker='o', label='Distance Step Improvement')
plt.plot(data['Time_ns'], -data['DescStep'], linestyle='-', color='red', marker='o', label='Descent Step Improvement')
plt.xlabel('Time (ns)')
plt.ylabel('Energy')
plt.title('Improvements made by global distance step and descent step over time')
plt.legend()
plt.grid(False)
plt.yscale("log")
plt.show()

