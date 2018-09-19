# --- Jose C. Garcia Alanis
# --- utf-8
# --- Python 3.6.2
#
# --- EEG prepossessing - DPX TT
# --- Version Jul 2018
#
# --- Artifact detection, interpolate bad channels, extract block data,
# --- filtering, re-referencing.

# =================================================================================================
# ------------------------------ Import relevant extensions ---------------------------------------
import glob
import os
import pandas as pd
import mne
import sys
# import matplotlib.pyplot as plt
# from mne.preprocessing import ICA
# from mne.preprocessing import create_eog_epochs

# ========================================================================
# --- GLOBAL SETTINGS
# --- SET PATH TO .bdf-files, summary files and output
data_path = '/Users/Josealanis/Documents/Experiments/fla_soc/eeg/fla_soc_bdf'
summary_path = '/Users/Josealanis/Documents/Experiments/fla_soc/eeg/fla_soc_mne_summary/'
output_path = '/Users/Josealanis/Documents/Experiments/fla_soc/eeg/fla_soc_mne_raws/'

# Channels to be ignored during artifact detection procedure
ignore_ch = {'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8'}
# Threshold for plotting
clip = None

# === LOOP THROUGH FILES AND RUN PRE-PROCESSING ==========================
for file in glob.glob(os.path.join(data_path, '*.bdf')):

    # --- 1) Set up paths and file names -----------------------
    filepath, filename = os.path.split(file)
    filename, ext = os.path.splitext(filename)
    print(filename)

    # --- 2) READ IN THE DATA ----------------------------------
    # Set EEG arrangement
    montage = mne.channels.read_montage(kind='biosemi64')
    # Import raw data
    raw = mne.io.read_raw_edf(file,
                              montage=montage,
                              preload=True,
                              stim_channel=-1,
                              exclude=['EOGH_rechts', 'EOGH_links',
                                       'EOGV_oben', 'EOGV_unten',
                                       'EXG1', 'EXG2',
                                       'EXG3', 'EXG4', 'EXG5', 'EXG6',
                                       'EXG7', 'EXG8'])

    # --- 3) EDIT DATA SET INFORMATION -------------------------
    # Note the sampling rate of recording
    sfreq = raw.info['sfreq']
    # and Buffer size ???
    bsize = raw.info['buffer_size_sec']

    # Channel names
    n_eeg = 64
    chans = raw.info['ch_names'][0:n_eeg]
    chans.extend(['Stim'])

    # Write a list of channel types (e.g., eeg, eog, ecg)
    chan_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                  'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                  'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                  'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                  'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                  'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                  'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                  'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                  'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                  'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                  'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                  'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                  'eeg', 'eeg', 'eeg', 'eeg', 'stim']

    # Bring it all together with MNE.function
    # for creating custom EEG info files
    info_custom = mne.create_info(chans, sfreq, chan_types, montage)

    # Add description / name of experiment
    info_custom['description'] = 'Flanker Task: Cooperation vs. Competition'

    # Replace the mne info structure with the customized one
    # which has the correct labels, channel types and positions.
    raw.info = info_custom
    raw.info['buffer_size_sec'] = bsize

    # Check data information
    print(raw.info)

    # --- 4) GET EVENT INFORMATION -----------------------------
    # Next, define the type of data you want to work with
    picks = mne.pick_types(raw.info,
                           meg=False,
                           eeg=True,
                           eog=True,
                           stim=True)

    # Get events
    events = mne.find_events(raw,
                             stim_channel='Stim',
                             output='onset',
                             min_duration=0.002)

    # --- 5) GET EVENTS REPRESENTING START AND END OF BLOCK ----
    # # Latency of starts and end markers
    # starts = events[events[:, 2] == 65663, ]
    # ends = events[events[:, 2] == 245, ]
    # print('There are', len(starts), 'starts and', len(ends), 'ends.')

    # --- 5) GET EVENT LATENCIES -------------------------------
    # Latency of target events
    evs = events[(events[:, 2] >= 70) & (events[:, 2] <= 75), ]
    latencies = events[(events[:, 2] >= 70) & (events[:, 2] <= 75), 0]

    # --- 6) SAVE START AND END OF EXP. BLOCKS -----------------
    # If 1248 events found, i.e., 48 practice + 3 x 400 for exp. conditions (solo, cooperation,
    # and competition). Then keep data from experimental conditions.
    if len(latencies) == 1248:
        # Save start and end latencies of blocks (+/- 2 secs).
        # Solo condition
        b1s = (latencies[48] / sfreq) - 2
        b1e = (latencies[447] / sfreq) + 2
        # First exp. condition
        b2s = (latencies[448] / sfreq) - 2
        b2e = (latencies[847] / sfreq) + 2
        # Second exp. condition
        b3s = (latencies[848] / sfreq) - 2
        b3e = (latencies[1247] / sfreq) + 2
        # Print summary
        print('Got', len(latencies), 'event latencies.')
    else:
        # If trials less than 1248 --> something is wrong.
        sys.exit('ERROR, number of trials not 1248; script stop.')

    # Print start and end of task blocks and duration.
    print('Solo block from', round(b1s, 3), 'to', round(b1e, 3), '/   Block length:',
          round(b1e - b1s, 3), 'sec.')
    print('First exp. condition from', round(b2s, 3), 'to', round(b2e, 3), '/   Block length:',
          round(b2e - b2s, 3), 'sec.')
    print('Second exp. condition from', round(b3s, 3), 'to', round(b3e, 3), '/   Block length:',
          round(b3e - b3s, 3), 'sec.')

    # --- 7) EXTRACT BLOCK DATA --------------------------------
    # Block 1
    raw_bl1 = raw.copy().crop(tmin=b1s, tmax=b1e)
    # Block 2
    raw_bl2 = raw.copy().crop(tmin=b2s, tmax=b2e)
    # Block 3
    raw_bl3 = raw.copy().crop(tmin=b3s, tmax=b3e)

    # Concatenate extracted blocks
    raw_blocks = mne.concatenate_raws([raw_bl1, raw_bl2, raw_bl3])

    # Find events in the blocks data set
    evs_blocks = mne.find_events(raw_blocks,
                                 stim_channel='Stim',
                                 output='onset',
                                 min_duration=0.002)

    # Number of events in blocks data set
    print(len(evs_blocks[(evs_blocks[:, 2] >= 70) &
                         (evs_blocks[:, 2] <= 75), 0]), 'events found.')

    # --- 8) APPLY FILTER TO DATA ------------------------------
    raw_blocks.filter(0.1, 50, fir_design='firwin')

    # --- 9) FIND DISTORTED SEGMENTS IN DATA -------------------
    # Copy of data
    x = raw_blocks.get_data()

    # Channels to be checked by artifact detection procedure
    ch_ix = [k for k in range(len(raw.info['ch_names'])) if
             raw.info['ch_names'][k] not in ignore_ch and k < n_eeg]

    # Detect artifacts (i.e., absolute amplitude > 500 microV)
    times = []
    annotations_df = pd.DataFrame(times)
    onsets = []
    duration = []
    for j in range(0, len(x[0])):
        if len(times) > 0:
            if j <= (times[-1] + int(2 * sfreq)):
                continue
        t = []
        for i in ch_ix:
            t.append(abs(x[i][j]))
        if max(t) >= 5e-4:
            times.append(float(j))
    # If artifact found create annotations for raw data
    if len(times) > 0:
        # Save onsets
        annotations_df = pd.DataFrame(times)
        annotations_df.columns = ['Onsets']
        # Include one second before artifact onset
        onsets = (annotations_df['Onsets'].values / sfreq) - 1
        # Merge with previous annotations
        duration = [2] * len(onsets) + list(raw_blocks.annotations.duration)
        labels = ['Bad'] * len(onsets) + list(raw_blocks.annotations.description)
        onsets = list(onsets)
        # Append onsets of previous annotations
        for i in range(0, len(list(raw_blocks.annotations.onset))):
            onsets.append(list(raw_blocks.annotations.onset)[i])
        # Create new annotation info
        annotations = mne.Annotations(onsets, duration, labels)
        raw_blocks.annotations = annotations

    # --- 10) RE-REFERENCE TO AVERAGE OF 64 ELECTRODES  -------
    raw_blocks.set_eeg_reference(ref_channels='average',
                                 projection=False)

    # ======================================================================
    # --- 11) CHECK FOR INCONSISTENCIES ------------------------------------
    # Alert
    os.system('say "Gute Reise"')
    # Plot
    raw_blocks.plot(n_channels=65,
                    scalings=dict(eeg=100e-6),
                    events=evs_blocks,
                    bad_color='red',
                    block=True,
                    clipping=clip)

    # Save bad channels
    bad_ch = raw_blocks.info['bads']

    # --- IF BAD CHANNELS FOUND:
    # --- INTERPOLATE CHANNELS, RE-RUN PRE-PROCESSING STEPS
    if len(bad_ch) >= 1:
        # Block 1
        raw_bl1 = raw.copy().crop(tmin=b1s, tmax=b1e)
        # Block 2
        raw_bl2 = raw.copy().crop(tmin=b2s, tmax=b2e)
        # Block 3
        raw_bl3 = raw.copy().crop(tmin=b3s, tmax=b3e)

        # Concatenate extracted blocks
        raw_blocks = mne.concatenate_raws([raw_bl1, raw_bl2, raw_bl3])
        # Find events in the concatenated data set
        evs_blocks = mne.find_events(raw_blocks,
                                     stim_channel='Stim',
                                     output='onset',
                                     min_duration=0.002)
        # Mark as bad
        raw_blocks.info['bads'] = bad_ch
        # INTERPOLATE BAD CHANNELS
        raw_blocks.interpolate_bads(reset_bads=True,
                                    verbose=False,
                                    mode='accurate')
        # Apply filter
        raw_blocks.filter(0.1, 50, fir_design='firwin')
        # Copy of data
        x = raw_blocks.get_data()

        # Channels to be checked by artifact detection procedure
        ch_ix = [k for k in range(len(raw.info['ch_names'])) if
                 raw.info['ch_names'][k] not in ignore_ch and k < n_eeg]

        # Detect artifacts (i.e., absolute amplitude > 500 microV)
        times = []
        annotations_df = pd.DataFrame(times)
        onsets = []
        duration = []
        for j in range(0, len(x[0])):
            if len(times) > 0:
                if j <= (times[-1] + int(2 * sfreq)):
                    continue
            t = []
            for i in ch_ix:
                t.append(abs(x[i][j]))
            if max(t) >= 5e-4:
                times.append(float(j))
        # If artifact found create annotations for raw data
        if len(times) > 0:
            # Save onsets
            annotations_df = pd.DataFrame(times)
            annotations_df.columns = ['Onsets']
            # Include one second before artifact onset
            onsets = (annotations_df['Onsets'].values / sfreq) - 1
            # Merge with previous annotations
            duration = [2] * len(onsets) + list(raw_blocks.annotations.duration)
            labels = ['Bad'] * len(onsets) + list(raw_blocks.annotations.description)
            onsets = list(onsets)
            # Append onsets of previous annotations
            for i in range(0, len(list(raw_blocks.annotations.onset))):
                onsets.append(list(raw_blocks.annotations.onset)[i])
            # Create new annotation info
            annotations = mne.Annotations(onsets, duration, labels)
            raw_blocks.annotations = annotations

        # --- RE-REFERENCE TO AVERAGE OF 64 ELECTRODES  -------
        raw_blocks.set_eeg_reference(ref_channels='average',
                                     projection=False)

        # --- PLOT TO CHECK
        # Alert
        os.system('say "Gute Reise"')
        # Plot
        raw_blocks.plot(n_channels=65,
                        scalings=dict(eeg=100e-6),
                        events=evs_blocks,
                        bad_color='red',
                        block=True,
                        clipping=clip)

    # --- 12) WRITE PRE-PROCESSING SUMMARY ---------------------
    name = str(filename) + '_summary'
    file = open(summary_path + '%s.txt' % name, 'w')
    # Number of Trials
    file.write('number of trials\n')
    file.write(str(len(evs_blocks[(evs_blocks[:, 2] >= 70) &
                                  (evs_blocks[:, 2] <= 75), 0])) + '\n')
    # Interpolated channels
    file.write('interpolated Channels\n')
    for ch in bad_ch:
        file.write('%s\n' % ch)
    # Artifacts detected
    file.write('annotated times\n')
    for on in onsets:
        file.write('%s\n' % str(on))
    # Total distorted time
    file.write('total annotated time (s)\n')
    file.write(str(sum(duration)))
    # Close file
    file.close()

    # --- 14) SAVE RAW FILE -----------------------------------
    # Pick electrode to use
    picks = mne.pick_types(raw_blocks.info,
                           meg=False,
                           eeg=True,
                           eog=False,
                           stim=True)

    # Save segmented data
    raw_blocks.save(output_path + filename + '-raw.fif',
                    picks=picks,
                    overwrite=True)
