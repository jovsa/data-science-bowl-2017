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

def get_patient_data_chunks(patient_id, output):
    print("patient_id", patient_id)

    output.put(patient_id)

def predict_features():

    random.seed(1234)
    output = mp.Queue()

    # Setup a list of processes that we want to run
    processes = [mp.Process(target=get_patient_data_chunks, args=(x, output)) for x in range(10)]

    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()

    # Get process results from the output queue
    results = [output.get() for p in processes]

    print(results)


    # for i in range(0, 10):
    #     get_patient_data_chunks(i)
    return

if __name__ == '__main__':
    start_time = time.time()
    print("There are %d CPUs on this machine" % mp.cpu_count())
    predict_features()

    end_time = time.time()
    print("Total Time usage: " + str(timedelta(seconds=int(round(end_time - start_time)))))
