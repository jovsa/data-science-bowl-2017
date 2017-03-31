import glob
import os
import re
import pandas as pd
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from datetime import timedelta
import sys
import datetime
import tensorflow as tf
import math
import multiprocessing as mp
import random

# Fixes "SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame"
pd.options.mode.chained_assignment = None

import scipy.misc
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def worker(patient_id):
    print("patient_id", patient_id)
    return patient_id

def predict_features():

    ids = [1,2,3,5,9,10,12]

    pool = mp.Pool(processes=4)
    results = [pool.apply(worker, args=(x,)) for x in range(0,10000000000000000)]
    print(results)
    return

if __name__ == '__main__':
    start_time = time.time()
    print("There are %d CPUs on this machine" % mp.cpu_count())
    predict_features()

    end_time = time.time()
    print("Total Time usage: " + str(timedelta(seconds=int(round(end_time - start_time)))))
