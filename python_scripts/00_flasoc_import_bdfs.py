# --- jose C. garcia alanis
# --- utf-8
# --- Python 3.7.3 / mne 0.18.1
#
# --- eeg pre-processing for flanker-social
# --- version: june 2019
#
# --- import data, crate info for file
# --- save to .fif

# ========================================================================
# ------------------- import relevant extensions -------------------------
import glob
import os

import re

from mne.channels import read_montage
from mne.io import read_raw_bdf
from mne import create_info

# ========================================================================
# --- global settings
# --- prompt user to set project path
root_path = input("Type path to project directory: ")

# path to eeg files
data_path = os.path.join(root_path)

if not os.path.exists(os.path.join(data_path, 'raws')):
    os.mkdir(os.path.join(data_path, 'raws'))

output_path = os.path.join(data_path, 'raws')

# files to be analysed
files = glob.glob(os.path.join(data_path, '*.bdf'))

# task info
task_description = 'flanker task - social video call'

# exclude channel on import
exclude = {'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8'}
# Channels to be ignored during artifact detection procedure
ignore_ch = {'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8'}

# Threshold for plotting
clip = None

# === LOOP THROUGH FILES AND RUN PRE-PROCESSING ==========================
for file in files:

    # --- 1) Set up paths and file names -----------------------
    filepath, filename = os.path.split(file)
    filename, ext = os.path.splitext(filename)

    subj = re.findall(r'\d+', filename)[0]

    # --- 2) READ IN THE DATA ----------------------------------
    # Set EEG arrangement
    montage = read_montage(kind='biosemi64')
    # Import raw data
    raw = read_raw_bdf(file,
                       montage=montage,
                       preload=True,
                       exclude=exclude)

    # Note the sampling rate of recording
    sfreq = raw.info['sfreq']
    # all channels in raw
    chans = raw.info['ch_names']
    # channels in montage
    montage_chans = montage.ch_names
    # nr of eeg channels
    n_eeg = len([chan for chan in chans if chan in montage_chans])
    # channel types
    types = []
    for chan in chans:
        if chan in montage_chans:
            types.append('eeg')
        elif re.match('EOG|EXG', chan):
            types.append('eog')
        else:
            types.append('stim')

    # create custom info for subj file
    custom_info = create_info(chans, sfreq, types, montage)

    # description / name of experiment
    custom_info['description'] = task_description

    # overwrite file info
    raw.info = custom_info

    # --- 4) set reference to remove residual line noise  ------
    raw.set_eeg_reference(['Cz'], projection=False)

    # sampling rate of recording
    sfreq = raw.info['sfreq']

    # --- 10) lower the sample rate  ---------------------------
    raw.resample(sfreq=512.)

    # save file
    raw.save(os.path.join(output_path, '%s_task-raw.fif' % subj),
             overwrite=True)
