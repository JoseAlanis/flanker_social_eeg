# --- Jose C. Garcia Alanis
# --- utf-8
# --- Python 3.6.2
#
# --- EEG prepossessing - DPX TT
# --- Version Sep 2018
#
# --- Extract epochs for trial and experimental conditions,
# --- save results

# =================================================================================================
# ------------------------------ Import relevant extensions ---------------------------------------
import glob
import os
import mne
from mne import io

# from mne.time_frequency import tfr_array_morlet, tfr_morlet
# from mne.preprocessing import ICA
# from mne.preprocessing import create_eog_epochs

# ========================================================================
# --- GLOBAL SETTINGS
# --- SET PATH TO .prunded-files, summary files and output
data_path = '/Users/Josealanis/Documents/Experiments/fla_soc/eeg/fla_soc_mne_pruned/'
summary_path = '/Users/Josealanis/Documents/Experiments/fla_soc/eeg/fla_soc_mne_summary/'
choice_output_path = '/Users/Josealanis/Documents/Experiments/fla_soc/eeg/fla_soc_mne_epochs/'


# === LOOP THROUGH FILES AND EXTRACT EPOCHS =========================
for file in glob.glob(os.path.join(data_path, '*.fif')):

    # --- 1) Set up paths and file names -----------------------
    filepath, filename = os.path.split(file)
    filename, ext = os.path.splitext(filename)
    print(filename)

    # --- 2) Read in the data ----------------------------------
    raw = io.read_raw_fif(file,
                          preload=True)

    # Check info
    print(raw.info)

    # --- 3) RECODE EVENTS -----------------------------------------
    #  Get events
    evs = mne.find_events(raw,
                          stim_channel='Stim',
                          output='onset',
                          min_duration=0.002)

    # Copy of events
    new_evs = evs.copy()

    # Global variables
    broken = []
    trial = 0
    # Recode reactions
    for i in range(new_evs[:, 2].size):
        if new_evs[:, 2][i] == 71:
            trial += 1
            next_t = new_evs[range(i, i + 3)]
            if len([k for k in list(next_t[:, 2]) if k in {101, 102, 201, 202}]) == 0:
                broken.append(trial)
                valid = False
                continue
            else:
                valid = True
                continue
        elif new_evs[:, 2][i] in {11, 12, 21, 22} and valid:
            if new_evs[:, 2][i] in {11, 12}:
                suffix = 1 # Congr.
            elif new_evs[:, 2][i] in {21, 22}:
                suffix = 2 # Incongr.
            continue
        # Check if event preceded by other reaction
        elif new_evs[:, 2][i] in {101, 102, 201, 202} and valid:
            if new_evs[:, 2][i] in [101, 102] and suffix == 1:
                new_evs[:, 2][i] = 91 # Correct Congr.
            elif new_evs[:, 2][i] in [101, 102] and suffix == 2:
                new_evs[:, 2][i] = 92 # Correct Incongr.
            elif new_evs[:, 2][i] in [201, 202] and suffix == 1:
                new_evs[:, 2][i] = 93 # Incorrect Congr.
            elif new_evs[:, 2][i] in [201, 202] and suffix == 2:
                new_evs[:, 2][i] = 94 # Incorrect Incongr.
            valid = False
            continue
        elif new_evs[:, 2][i] in {101, 102, 201, 202} and not valid:
            continue

    # --- 4) PICK CHANNELS TO SAVE -----------------------------
    picks = mne.pick_types(raw.info,
                           meg=False,
                           eeg=True,
                           eog=False,
                           stim=False)

    # --- 5) EXTRACT EPOCHS ------------------------------------
    # Choices
    choice_evs = new_evs[(new_evs[:, 2] == 91) | (new_evs[:, 2] == 92) |
                         (new_evs[:, 2] == 93) | (new_evs[:, 2] == 94), :]

    # Set event ids
    choice_event_id = {'Correct congr.': 91,
                       'Correct incongr.': 92,
                       'Incorrect congr.': 93,
                       'Incorrect incongr.': 94}

    # Extract choice epochs
    choice_epochs = mne.Epochs(raw, choice_evs, choice_event_id,
                               on_missing='ignore',
                               tmin=-1,
                               tmax=1,
                               baseline=(-.300, -.100),
                               preload=True,
                               reject_by_annotation=True,
                               picks=picks)
    # Clean epochs
    clean_choices = choice_epochs.selection+1
    bads = [x for x in set(list(range(0, trial - len(broken)))) if x not in set(choice_epochs.selection)]
    bads = [x+1 for x in bads]

    # --- 6) WRITE SUMMARY -------------------------------------
    # Write file
    name = filename.split('_')[0] + '_epochs_summary'
    sum_file = open(summary_path + '%s.txt' % name, 'w')

    sum_file.write('Broken epochs are ' + str(len(broken)) + ':\n')
    for b in broken:
        sum_file.write('%s \n' % b)

    sum_file.write('Rejected epochs are ' + str(len(bads)) + ':\n')
    for a in bads:
        sum_file.write('%s \n' % a)

    sum_file.write('Extracted epochs are ' + str(len(clean_choices)) + ':\n')
    for e in clean_choices:
        sum_file.write('%s \n' % e)

    sum_file.close()

    choice_epochs.save(choice_output_path + filename.split('_')[0] + '_flasoc-epo.fif')
