from scipy.signal import butter, filtfilt
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from pandas import DataFrame, Series

fsamp = 250
tsample = 1 / fsamp
f_low = 50
f_high = 1
order = 2
channel_vector = [1,2, 3, 4, 5]
n_ch = len(channel_vector)
df = pd.read_csv("/home/runge/openbci/application.linux64/application.linux64/OpenBCI-RAW-right_strait_up_new.txt")
df = df[channel_vector].dropna(axis=0)

processed_signal = df.copy()

b, a = butter(order, (order * f_low * 1.0) / fsamp * 1.0, btype='low')
for i in range(0, n_ch):
    processed_signal.ix[:, i] = np.transpose(filtfilt(b, a, df.ix[:, i]))

b1, a1 = butter(order, (order * f_high * 1.0) / fsamp * 1.0, btype='high')
for i in range(0, n_ch):
    processed_signal.ix[:, i] = np.transpose(filtfilt(b1, a1, processed_signal.ix[:, i]))

Wn = (np.array([58.0, 62.0]) / 500 * order).tolist()
b3, a3 = butter(order, Wn, btype='stop')
for i in range(0, n_ch):
    processed_signal.ix[:, i] = np.transpose(filtfilt(b3, a3, processed_signal.ix[:, i]))

start = 520
end = 620

plt.figure(figsize=(12, 8))
plt.subplot(511)
plt.plot(processed_signal.ix[:, 0][start * fsamp:end * fsamp])
plt.subplot(512)
plt.plot(processed_signal.ix[:, 1][start * fsamp:end * fsamp])
plt.subplot(513)
plt.plot(processed_signal.ix[:, 2][start * fsamp:end * fsamp])
plt.subplot(514)
plt.plot(processed_signal.ix[:, 3][start * fsamp:end * fsamp])
plt.subplot(515)
plt.plot(processed_signal.ix[:, 4][start * fsamp:end * fsamp])
plt.show()

print "---"
