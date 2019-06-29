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
import os
import glob
import re

from mne import pick_types, Epochs, combine_evoked
from mne.viz import plot_compare_evokeds
from mne.io import read_raw_fif

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
data_path = os.path.join(root_path, 'derivatives/pruned')
# output path
output_path = os.path.join(root_path, 'derivatives/segmentation')

# create directory for save
if not os.path.isdir(os.path.join(output_path)):
    os.mkdir(os.path.join(output_path))

# files to be analysed
files = glob.glob(os.path.join(data_path, '*_pruned-raw.fif'))


# === LOOP THROUGH FILES AND EXTRACT EPOCHS =========================
for file in files:

    # --- 1) set up paths and file names -----------------------
    filepath, filename = os.path.split(file)
    # subject in question
    subj = re.findall(r'\d+', filename)[0]

    # --- 2) Read in the data ----------------------------------
    raw = read_raw_fif(file, preload=True)

    # --- 3) RECODE EVENTS -----------------------------------------
    #  Get events
    evs = mne.find_events(raw,
                          stim_channel='Status',
                          output='onset',
                          min_duration=0.002)

    # Copy of events
    new_evs = evs.copy()

    # Global variables
    broken = []
    trial = 0
    # Recode reactions
    for i in range(len(new_evs[:, 2])):
        if new_evs[:, 2][i] == 71:
            next_t = new_evs[range(i, i + 3)]
            if [k for k in list(next_t[:, 2]) if k in {101, 102, 201, 202}]:
                valid = True
                trial += 1
                continue
            else:
                broken.append(trial)
                valid = False
                trial += 1
                continue

        elif new_evs[:, 2][i] in {11, 12, 21, 22}:
            if new_evs[:, 2][i] in {11, 12}:
                suffix = 1 # Congr.
            elif new_evs[:, 2][i] in {21, 22}:
                suffix = 2 # Incongr.
            continue
        # Check if event preceded by other reaction

        elif new_evs[:, 2][i] in {101, 102, 201, 202} and valid:
            if trial <= 48:
                if new_evs[:, 2][i] in [101, 102] and suffix == 1:
                    new_evs[:, 2][i] = 1091  # Correct Congr.
                elif new_evs[:, 2][i] in [101, 102] and suffix == 2:
                    new_evs[:, 2][i] = 1092  # Correct Incongr.
                elif new_evs[:, 2][i] in [201, 202] and suffix == 1:
                    new_evs[:, 2][i] = 1093  # Incorrect Congr.
                elif new_evs[:, 2][i] in [201, 202] and suffix == 2:
                    new_evs[:, 2][i] = 1094  # Incorrect Incongr.
                valid = False
                continue
            elif trial <= 448:
                if new_evs[:, 2][i] in [101, 102] and suffix == 1:
                    new_evs[:, 2][i] = 2091  # Correct Congr.
                elif new_evs[:, 2][i] in [101, 102] and suffix == 2:
                    new_evs[:, 2][i] = 2092  # Correct Incongr.
                elif new_evs[:, 2][i] in [201, 202] and suffix == 1:
                    new_evs[:, 2][i] = 2093  # Incorrect Congr.
                elif new_evs[:, 2][i] in [201, 202] and suffix == 2:
                    new_evs[:, 2][i] = 2094  # Incorrect Incongr.
                valid = False
                continue
            elif trial <= 848:
                if new_evs[:, 2][i] in [101, 102] and suffix == 1:
                    new_evs[:, 2][i] = 3091  # Correct Congr.
                elif new_evs[:, 2][i] in [101, 102] and suffix == 2:
                    new_evs[:, 2][i] = 3092  # Correct Incongr.
                elif new_evs[:, 2][i] in [201, 202] and suffix == 1:
                    new_evs[:, 2][i] = 3093  # Incorrect Congr.
                elif new_evs[:, 2][i] in [201, 202] and suffix == 2:
                    new_evs[:, 2][i] = 3094  # Incorrect Incongr.
                valid = False
                continue
            elif trial <= 1248:
                if new_evs[:, 2][i] in [101, 102] and suffix == 1:
                    new_evs[:, 2][i] = 4091  # Correct Congr.
                elif new_evs[:, 2][i] in [101, 102] and suffix == 2:
                    new_evs[:, 2][i] = 4092  # Correct Incongr.
                elif new_evs[:, 2][i] in [201, 202] and suffix == 1:
                    new_evs[:, 2][i] = 4093  # Incorrect Congr.
                elif new_evs[:, 2][i] in [201, 202] and suffix == 2:
                    new_evs[:, 2][i] = 4094  # Incorrect Incongr.
                valid = False
                continue

        elif new_evs[:, 2][i] in {101, 102, 201, 202} and not valid:
            continue

    # --- 4) PICK CHANNELS TO SAVE -----------------------------
    picks = pick_types(raw.info,
                       meg=False,
                       eeg=True,
                       eog=False,
                       stim=False)

    # --- 5) EXTRACT EPOCHS ------------------------------------
    # set event ids
    choice_event_id = {'P_Correct congr.': 1091,
                       'P_Correct incongr.': 1092,
                       'P_Incorrect congr.': 1093,
                       'P_Incorrect incongr.': 1094,
                       'S_Correct congr.': 2091,
                       'S_Correct incongr.': 2092,
                       'S_Incorrect congr.': 2093,
                       'S_Incorrect incongr.': 2094,
                       '+_Correct congr.': 3091,
                       '+_Correct incongr.': 3092,
                       '+_Incorrect congr.': 3093,
                       '+_Incorrect incongr.': 3094,
                       '-_Correct congr.': 4091,
                       '-_Correct incongr.': 4092,
                       '-_Incorrect congr.': 4093,
                       '-_Incorrect incongr.': 4094
                       }

    # Extract choice epochs
    choice_epochs = Epochs(raw, new_evs, choice_event_id,
                           on_missing='ignore',
                           tmin=-1,
                           tmax=1,
                           baseline=(-.400, -.150),
                           preload=True,
                           reject_by_annotation=True,
                           picks=picks)


    choice_epochs['S_Incorrect incongr.'].average().plot_joint()

    keys = {#'P_Correct congr.',
            #'P_Correct incongr.',
            #'P_Incorrect congr.',
            #'P_Incorrect incongr.',
            #'S_Correct congr.',
            'S_Correct incongr.',
            #'S_Incorrect congr.',
            'S_Incorrect incongr.',
           # '+_Correct congr.',
            '+_Correct incongr.',
            #'+_Incorrect congr.',
            '+_Incorrect incongr.',
            #'-_Correct congr.',
            '-_Correct incongr.',
            #'-_Incorrect congr.',
            '-_Incorrect incongr.'}

    evokeds = {key: choice_epochs[key].average() for key in keys}

    pick = evokeds['S_Correct incongr.'].ch_names.index('Cz')
    plot_compare_evokeds(evokeds,  picks=pick, ylim=dict(eeg=[20, -40]))


    diff_wave_solo = combine_evoked([-evokeds['S_Correct incongr.'],
                                     evokeds['S_Incorrect incongr.']],
                                    weights="equal")

    diff_wave_pos = combine_evoked([-evokeds['+_Correct incongr.'],
                                    evokeds['+_Incorrect incongr.']],
                                   weights="equal")

    diff_wave_neg = combine_evoked([-evokeds['-_Correct incongr.'],
                                    evokeds['-_Incorrect incongr.']],
                                   weights="equal")

    diff_waves = dict(solo=diff_wave_solo,
                      positive=diff_wave_pos,
                      negative=diff_wave_neg)

    plot_compare_evokeds(diff_waves,  picks=pick, ylim=dict(eeg=[10, -20]))


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
