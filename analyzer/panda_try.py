import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt

s = Series([1,2,3,4])
df =  pd.read_csv("/home/runge/openbci/OpenBCI_Python/build/dataset/OpenBCI-RAW-2017-04-06_15-10-11.csv")
df.plot()

print "---"


def compute_LLE(self, ):
    self.LLE = []
    for k in range(0, self.frames):
        current_frame = self.get_current_frame(k)
        
        self.LLE.append(m)

    if self.is_raw_data:
        self.LLE = np.asarray(self.LLE)
    else:
        self.LLE = np.asarray(self.LLE)[0]


def get_LLE(self):
    return self.LLE