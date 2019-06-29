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

import numpy as np
import pandas as pd

from mne.io import read_raw_fif
from mne import pick_types, Annotations

# ========================================================================
# --- global settings
# prompt user to set project path
root_path = input("Type path to project directory: ")

# look for directory
if os.path.isdir(root_path):
    print('Setting "root_path" to ', root_path)
else:
    raise NameError('Directory not found!')

# path to eeg files
data_path = os.path.join(root_path, 'raws')
# path for output
derivatives_path = os.path.join(root_path, 'derivatives')

# create directory for save
if not os.path.isdir(os.path.join(derivatives_path)):
    os.mkdir(os.path.join(derivatives_path))

if not os.path.isdir(os.path.join(derivatives_path, 'artifact_rejection')):
    os.mkdir(os.path.join(derivatives_path, 'artifact_rejection'))

output_path = os.path.join(derivatives_path, 'artifact_rejection')

# files to be analysed
files = glob.glob(os.path.join(data_path, '*-raw.fif'))

# ========================================================================
# ------------ loop through files and extract blocks  --------------------
for file in files:

    # --- 1) set up paths and file names -----------------------
    filepath, filename = os.path.split(file)
    # subject in question
    subj = re.findall(r'\d+', filename)[0]

    # --- 2) import the preprocessed data ----------------------
    raw = read_raw_fif(file, preload=True)

    # index of eeg channels
    picks_eeg = pick_types(raw.info, eeg=True, eog=False, stim=False)

    # index of eogs and stim channels
    picks_no_eeg = pick_types(raw.info, eeg=False, eog=True, stim=True)

    # channel names
    channels = raw.info['ch_names']
    # sampling frequency
    sfreq = raw.info['sfreq']

    # channels that should be ignored during the artifact detection procedure
    ignore_ch = {'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8'}
    # update dict
    ignore_ch.update({raw.info['ch_names'][chan] for chan in picks_no_eeg})

    # --- 4.1) filter the data ---------------------------------
    # copy the file
    raw_copy = raw.copy()
    # apply filter
    raw_copy.filter(l_freq=0.1, h_freq=50, picks=['eeg', 'eog'],
                    filter_length='auto',
                    l_trans_bandwidth='auto', h_trans_bandwidth='auto',
                    method='fir', phase='zero', fir_window='hamming',
                    fir_design='firwin')

    # --- 4.2) find distorted segments in data -----------------
    # copy of data
    data = raw_copy.get_data(picks_eeg)

    # channels to be checked by artifact detection procedure
    ch_ix = [raw_copy.info['ch_names'].index(chan) for chan in
             raw_copy.info['ch_names'] if chan not in ignore_ch]  # noqa

    # detect artifacts (i.e., absolute amplitude > 500 microV)
    times = []
    annotations_df = pd.DataFrame(times)
    onsets = []
    duration = []
    annotated_channels = []
    bad_chans = []

    # loop through samples
    for sample in range(0, data.shape[1]):
        if len(times) > 0:
            if sample <= (times[-1] + int(1 * sfreq)):
                continue
        peak = []
        for channel in ch_ix:
            peak.append(abs(data[channel][sample]))
        if max(peak) >= 400e-6:
            times.append(float(sample))
            annotated_channels.append(channels[ch_ix[int(np.argmax(peak))]])
    # If artifact found create annotations for raw data
    if len(times) > 0:
        # Save onsets
        annotations_df = pd.DataFrame(times)
        annotations_df.columns = ['Onsets']
        # Include one second before artifact onset
        onsets = (annotations_df['Onsets'].values / sfreq) - 1
        # Merge with previous annotations
        duration = [2] * len(onsets) + list(raw_copy.annotations.duration)
        labels = ['Bad'] * len(onsets) + list(
            raw_copy.annotations.description)
        onsets = list(onsets)
        # Append onsets of previous annotations
        for i in range(0, len(list(raw_copy.annotations.onset))):
            onsets.append(list(raw_copy.annotations.onset)[i])
        # Create new annotation info
        annotations = Annotations(onsets, duration, labels)
        raw_copy.set_annotations(annotations)

    # save frequency of annotation per channel
    frequency_of_annotation = {x: annotated_channels.count(x) for x in
                               annotated_channels}  # noqa
    # if exceeds 0.9% of total time --> mark as bad channel
    threshold = (raw_copy._last_time - raw_copy._first_time) * .009
    # save bads in info structure
    bad_chans = [chan for chan, value in frequency_of_annotation.items() if
                 value >= int(threshold)]  # noqa
    raw.info['bads'] = bad_chans

    # --- if bad channels were found, repeat preprocessing ---------
    if bad_chans:
        # copy the file
        raw_copy = raw.copy()
        # interpolate bads
        raw_copy.interpolate_bads(reset_bads=True,
                                  verbose=False,
                                  mode='accurate')
        # apply filter
        raw_copy.filter(l_freq=0.1, h_freq=50, picks=['eeg', 'eog'],
                        filter_length='auto',
                        l_trans_bandwidth='auto', h_trans_bandwidth='auto',
                        method='fir', phase='zero', fir_window='hamming',
                        fir_design='firwin')

        # --- find distorted segments in data ----------------------
        # copy of data
        data = raw_copy.get_data(picks_eeg)

        # channels to be checked by artifact detection procedure
        ch_ix = [raw_copy.info['ch_names'].index(chan) for chan in
                 raw_copy.info['ch_names'] if chan not in ignore_ch]

        # detect artifacts (i.e., absolute amplitude > 500 microV)
        times = []
        annotations_df = pd.DataFrame(times)
        onsets = []
        duration = []
        annotated_channels = []
        bad_chans = []

        # loop through samples
        for sample in range(0, data.shape[1]):
            if len(times) > 0:
                if sample <= (times[-1] + int(1 * sfreq)):
                    continue
            peak = []
            for channel in ch_ix:
                peak.append(abs(data[channel][sample]))
            if max(peak) >= 400e-6:
                times.append(float(sample))
                annotated_channels.append(channels[ch_ix[int(np.argmax(peak))]])
        # If artifact found create annotations for raw data
        if len(times) > 0:
            # Save onsets
            annotations_df = pd.DataFrame(times)
            annotations_df.columns = ['Onsets']
            # Include one second before artifact onset
            onsets = (annotations_df['Onsets'].values / sfreq) - 1
            # Merge with previous annotations
            duration = [2] * len(onsets) + list(raw_copy.annotations.duration)
            labels = ['Bad'] * len(onsets) + list(
                raw_copy.annotations.description)
            onsets = list(onsets)
            # Append onsets of previous annotations
            for i in range(0, len(list(raw_copy.annotations.onset))):
                onsets.append(list(raw_copy.annotations.onset)[i])
            # Create new annotation info
            annotations = Annotations(onsets, duration, labels)
            raw_copy.set_annotations(annotations)

    # --- 5) plot data and check for inconsistencies  ----------
    raw_copy.plot(scalings=dict(eeg=50e-6),
                  n_channels=len(raw.info['ch_names']),
                  bad_color='red',
                  block=True)

    # --- 6) RE-REFERENCE TO AVERAGE OF 64 ELECTRODES  ---------
    raw_copy.set_eeg_reference(ref_channels='average',
                               projection=False)

    # --- 7) save segmented data  -----------------------------
    # save file
    raw_copy.save(
        os.path.join(output_path, '%s_artifact_rejection-raw.fif' % subj),
        overwrite=True)

    # write summary
    name = '%s_artifact_detection' % subj
    sfile = open(os.path.join(output_path, '%s.txt' % name), 'w')
    # channels info
    sfile.write('Channels interpolated:\n')
    for ch in bad_chans:
        sfile.write('%s\n' % ch)
    # frequency of annotation
    sfile.write('Frequency of annotation:\n')
    for ch, f in frequency_of_annotation.items():
        sfile.write('%s, %f\n' % (ch, f))
    sfile.write('total annotated:\n')
    sfile.write(str(
        round(sum(duration) / (raw_copy._last_time - raw_copy._first_time),
              3)) + ' %\n')  # noqa
    sfile.close()

    del raw, raw_copy
