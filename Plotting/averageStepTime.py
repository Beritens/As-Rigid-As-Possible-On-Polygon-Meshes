import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

data = pd.read_csv('../measurements/measurements.csv', header=None, delimiter=",")
data.columns = ['Index', 'Time_ns', 'DistantTime', 'DescentTime', 'Energy', 'DistStep', 'DescStep']

time_differences = data['Time_ns'].diff()

time_differences = time_differences[1:]

average_time_difference = time_differences.mean()

print(f"Average time difference: {average_time_difference} ns")