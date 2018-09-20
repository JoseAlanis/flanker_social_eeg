# --- Jose C. Garcia Alanis
# --- utf-8
# --- Python 3.6.2
#
# --- EEG prepossessing - DPX TT
# --- Version Sep 2018
#
# --- Apply baseline, crop for smaller file, and
# ---Export epochs to .txt

# =================================================================================================
# ------------------------------ Import relevant extensions ---------------------------------------
import mne
import glob
import os

# ========================================================================
# --- GLOBAL SETTINGS
# --- SET PATH TO .epoch-files and output
input_path = '/Users/Josealanis/Documents/Experiments/fla_soc/eeg/fla_soc_mne_epochs/'
output_path = '/Users/Josealanis/Documents/Experiments/fla_soc/eeg/epochs_for_r/'

# === LOOP THROUGH FILES AND EXPORT EPOCHS ===============================
for file in glob.glob(os.path.join(input_path, '*-epo.fif')):

    filepath, filename = os.path.split(file)
    filename, ext = os.path.splitext(filename)
    name = filename.split('_')[0] + '_flasoc_choice_epo'

    # Read epochs
    epochs = mne.read_epochs(file, preload=True)
    # Apply baseline
    epochs.apply_baseline(baseline=(-0.3, -0.1))
    # Only keep time window fro -.3 to .99 sec. around motor response
    small = epochs.copy().crop(tmin=-.3, tmax=.99)

    # Transform to data frame
    epo = small.to_data_frame()
    # Round values
    epo = epo.round(3)
    # Export data frame
    epo.to_csv(output_path + name + '.txt', index=True)
