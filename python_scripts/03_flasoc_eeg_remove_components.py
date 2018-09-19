# --- Jose C. Garcia Alanis
# --- utf-8
# --- Python 3.6.2
#
# --- EEG prepossessing - DPX TT
# --- Version Sep 2018
#
# --- Inspect ICA components, remove bad components,
# --- save results

# =================================================================================================
# ------------------------------ Import relevant extensions ---------------------------------------
import glob
import os
import mne
from mne import io

# ========================================================================
# --- GLOBAL SETTINGS
# --- SET PATH TO .bdf-files, summary files and output
data_path = '/Users/Josealanis/Documents/Experiments/fla_soc/eeg/fla_soc_mne_raws/'
ica_path = '/Users/Josealanis/Documents/Experiments/fla_soc/eeg/fla_soc_mne_ica/'
summary_path = '/Users/Josealanis/Documents/Experiments/fla_soc/eeg/fla_soc_ica_summary/'
pruned_path = '/Users/Josealanis/Documents/Experiments/fla_soc/eeg/fla_soc_mne_pruned/'


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

    # --- 3) Get event information -----------------------------
    #  Get events
    evs = mne.find_events(raw,
                          stim_channel='Stim',
                          output='onset',
                          min_duration=0.002)

    # --- 4) Import ICA weights --------------------------------
    ica = mne.preprocessing.read_ica(ica_path + filename.split('-')[0] + '-ica.fif')

    # --- 5) Plot components time series -----------------------
    # Select bad components for rejection
    ica.plot_sources(raw, title=str(filename),
                     exclude=None,
                     picks=range(0, 25),
                     block=True)

    # Save bad components
    bad_comps = ica.exclude.copy()

    # --- 4) Remove bad components -----------------------------
    ica.apply(raw)

    # --- 5) Remove pruned data --------------------------------
    # Plot to check data
    clip = None
    raw.plot(n_channels=66, title=str(filename),
             scalings=dict(eeg=100e-6),
             events=evs,
             bad_color='red',
             clipping=clip,
             block=True)

    # --- 6) Write summary about removed components ------------
    name = str(filename.split('-')[0] ) + '_ica_summary'
    file = open(summary_path + '%s.txt' % name, 'w')
    # Number of Trials
    file.write('bad components\n')
    for cp in bad_comps:
        file.write('%s\n' % cp)
    # Close file
    file.close()

    # --- 7) Save raw file -----------------------------------
    # Pick electrode to use
    picks = mne.pick_types(raw.info,
                           meg=False,
                           eeg=True,
                           eog=False,
                           stim=True)

    # Save pruned data
    raw.save(pruned_path + filename.split('-')[0] + '_pruned-raw.fif',
             picks=picks,
             overwrite=True)
