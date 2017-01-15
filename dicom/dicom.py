import dicom
import glob
import numpy as np
import math

train_data_folder = '/kaggle/dev/data-science-bowl-2017-data/stage1/'

def get_slice_location(dcm):
    return float(dcm[0x0020, 0x1041].value)

# Returns a list of images for that patient_id, in ascending order of Slice Location
def load_patient(patient_id):
    path = train_data_folder + '{}/*.dcm'.format(patient_id)
    files = glob.glob(path)
    imgs = {}
    for f in files:
        dcm = dicom.read_file(f)
        img = dcm.pixel_array
        img[img == -2000] = 0
        sl = get_slice_location(dcm)
        imgs[sl] = img
        
    # Not a very elegant way to do this
    sorted_imgs = [x[1] for x in sorted(imgs.items(), key=lambda x: x[0])]
    return sorted_imgs

def get_patient_data(patient_ids):
    patient_data = {}
    for patient_id in patient_ids:
        pat = load_patient(patient_id)
        np_pat = np.array(pat)
        indices = range(0, len(pat), 10)
        patient_data[patient_id] = np_pat[indices]
    return patient_data
