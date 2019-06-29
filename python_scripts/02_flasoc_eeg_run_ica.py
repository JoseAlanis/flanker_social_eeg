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

from mne import pick_types
from mne.io import read_raw_fif
from mne.preprocessing import ICA


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
data_path = os.path.join(root_path, 'derivatives/artifact_rejection')
# output path
output_path = os.path.join(root_path, 'derivatives/ica')

# create directory for save
if not os.path.isdir(os.path.join(output_path)):
    os.mkdir(os.path.join(output_path))

# files to be analysed
files = glob.glob(os.path.join(data_path, '*-raw.fif'))

# === LOOP THROUGH FILES AND RUN PRE-PROCESSING ==========================
for file in files:

    # --- 1) set up paths and file names -----------------------
    filepath, filename = os.path.split(file)
    # subject in question
    subj = re.findall(r'\d+', filename)[0]

    # --- 2) READ IN THE DATA ----------------------------------
    # import preprocessed data.
    raw = read_raw_fif(file, preload=True)

    # --- 2) ICA DECOMPOSITION --------------------------------
    # ICA parameters
    n_components = 25
    method = 'infomax'
    fit_params = dict(extended=True)
    # decim = None
    reject = dict(eeg=4e-4)

    # Pick electrodes to use
    picks = pick_types(raw.info,
                       meg=False,
                       eeg=True,
                       eog=False,
                       stim=False)

    # ICA parameters
    ica = ICA(n_components=n_components,
              method=method,
              fit_params=fit_params)

    # Fit ICA
    ica.fit(raw.copy().filter(1, 50),
            picks=picks,
            reject=reject)

    ica.save(os.path.join(output_path, '%s_ica.fif' % subj))
    # --- 3) PLOT RESULTING COMPONENTS ------------------------
    # Plot components
    ica_fig = ica.plot_components(picks=range(0, 25), show=False)
    ica_fig.savefig(os.path.join(output_path, '%s_ica.pdf' % subj))
