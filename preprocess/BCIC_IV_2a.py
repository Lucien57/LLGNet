import numpy as np
import pandas as pd
import mne
import os
import lmdb
import pickle
from scipy.io import loadmat

'''
sample rate: 250Hz
electrodes: 25 channels(10-20 system) 22 EEG + 3 EOG(EOG-left, EOG-central, EOG-right)
bandpass filter: 0.5-100Hz
notch filter: 50Hz

T for training, E for evaluation
-2 minutes: eyes open
-1 minutes: eyes closed
-1 minutes: eyes movement

-276 0x0114 Idling EEG (eyes open)
-277 0x0115 Idling EEG (eyes closed)
-768 0x0300 Start of a trial
-769 0x0301 Cue onset left hand(class 1)
-770 0x0302 Cue onset right hand(class 2)
-771 0x0303 Cue onset foot (class 3)
-772 0x0304 Cue onset tongue (class 4)
-783 0x030F Cue unknown
-1023 0x03FF Rejected trial
-1072 0x0430 Eye movements
-32766 0x7FFE Start of a new run
'''
mne.set_log_level(verbose='ERROR') 
root_dir = 'data/BCIC_IV_2a/'
db = lmdb.open('lmdb/BCIC-IV-2a/', map_size=1024**3)
keys = []

bandpass = (0.5, 40.0)
resample_rate = 200.0
epoch_window = (0.0, 4.0)
preprocess_info = {
    "bandpass": list(bandpass),
    "resample": resample_rate,
    "tmin": epoch_window[0],
    "tmax": epoch_window[1],
}

subject_trial_counter = {}

for subject_id in range(1, 10):
    subject_trial_counter.setdefault(subject_id, 0)
    for session in ['T', 'E']:
        # data loading
        data_path = os.path.join(root_dir, 'data', f'A{subject_id:02d}{session}.gdf')
        raw = mne.io.read_raw_gdf(data_path, preload=True)
        # label loading
        label_path = os.path.join(root_dir, 'label', f'A{subject_id:02d}{session}.mat')
        label_file = loadmat(label_path)
        labels = label_file['classlabel'].squeeze() # (n_epochs,) value between 1 to 4
        # drop EOG channels
        raw.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])
        # bandpass filter
        raw.filter(preprocess_info['bandpass'][0], preprocess_info['bandpass'][1])
        # notch filter
        # if preprocess_info['bandpass'][0] <= 50 and 50 <= preprocess_info['bandpass'][1]:
            # raw.notch_filter(freqs = 50)
        # reference
        # raw.set_eeg_reference('average')
        # resample
        # raw.resample(preprocess_info['resample'])
        # event
        if session == 'T':
            events, event_id = mne.events_from_annotations(raw, event_id={'769': 1, '770': 2, '771': 3, '772': 4})
        elif session == 'E':
            events, event_id = mne.events_from_annotations(raw, event_id={'783': 5})
        # epoching
        epochs = mne.Epochs(raw, events, event_id, tmin=preprocess_info['tmin'], tmax=preprocess_info['tmax']-1.0/raw.info['sfreq'], baseline=None, preload=True)
        data = epochs.get_data()  # (n_epochs, n_channels, n_times)
        
        sfreq = raw.info['sfreq']
        for i, (data_idx, label_idx) in enumerate(zip(data, labels)):
            event_sample = int(epochs.events[i, 0]) if hasattr(epochs, "events") else i
            event_time = float(event_sample) / float(sfreq) if sfreq else float(i)
            trial_order = subject_trial_counter[subject_id]
            subject_trial_counter[subject_id] += 1
            key_bytes = f'Subject{subject_id}_Session{session}_Epoch{i}'.encode()
            data_dict = {
                'data': data_idx.astype(np.float32, copy=False),
                'label': int(label_idx) - 1,
                'subject': subject_id,
                'session': session,
                'epoch': i,
                'event_sample': event_sample,
                'event_time': event_time,
                'trial_order': trial_order,
            }
            txn = db.begin(write=True)
            txn.put(key_bytes, pickle.dumps(data_dict))
            txn.commit()
            keys.append(key_bytes)

txn = db.begin(write=True)
txn.put(b'__keys__', pickle.dumps(keys))
txn.put(b'__meta__', pickle.dumps({"preprocess": preprocess_info}))
txn.commit()
db.close()
