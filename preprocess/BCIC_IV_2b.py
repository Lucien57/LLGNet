import numpy as np
import pandas as pd
import mne
import os
import lmdb
import pickle
from scipy.io import loadmat

'''
sample rate: 250Hz
bandpass filter: 0.5-100Hz
notch filter: 50Hz

T for training, E for evaluation
-2 minutes: eyes open
-1 minutes: eyes closed
-1 minutes: eyes movement

-276 0x0114 Idling EEG (eyes open)
-277 0x0115 Idling EEG (eyes closed)
-768 0x0300 Start of a trial
-769 0x0301 Cue onset left (class 1)
-770 0x0302 Cue onset right (class 2)
-781 0x030D BCI feedback (continuous)
-783 0x030F Cue unknown
-1023 0x03FF Rejected trial
-1077 0x0435 Horizontal eye movement
-1078 0x0436 Vertical eye movement
-1079 0x0437 Eye rotation
-1081 0x0439 Eye blinks
-32766 0x7FFE Start of a new run
'''
mne.set_log_level(verbose='ERROR') 
root_dir = 'data/BCIC_IV_2b/'
db = lmdb.open('lmdb/BCIC-IV-2b/', map_size=512 *1024**2)
keys = []

bandpass = (1.0, 40.0)
resample_rate = 200.0
epoch_window = (0.0, 4.0)
preprocess_info = {
    "bandpass": list(bandpass),
    "resample": resample_rate,
    "tmin": epoch_window[0],
    "tmax": epoch_window[1],
}

# BCIC IV 2b has 9 subjects with 5 sessions each: first 3 are training (T), last 2 are evaluation (E)
for subject_id in range(1, 10):
    for session_num in ['01T', '02T', '03T', '04E', '05E']:
        session = 'T' if session_num in ['01T','02T','03T'] else 'E'  # Mark as training session
        data_path = os.path.join(root_dir, 'data', f'B{subject_id:02d}{session_num}.gdf')
        raw = mne.io.read_raw_gdf(data_path, preload=True)

        # label loading - for training sessions, labels are available
        label_path = os.path.join(root_dir, 'label', f'B{subject_id:02d}{session_num}.mat')
        label_file = loadmat(label_path)
        labels = label_file['classlabel'].squeeze()

        # drop EOG channels
        raw.drop_channels(['EOG:ch01', 'EOG:ch02', 'EOG:ch03'])
        # Bandpass filter 
        # raw.filter(preprocess_info['bandpass'][0], preprocess_info['bandpass'][1])
        # Reference - use average reference
        # raw.set_eeg_reference('average')
        # Resample to 100Hz for consistency
        # raw.resample(preprocess_info['resample'])

        # Event
        if session == 'T':
            events, event_id = mne.events_from_annotations(raw, event_id={'769': 1, '770': 2})
        else:
            events, event_id = mne.events_from_annotations(raw, event_id={'783': 3})

        # Epoching
        epochs = mne.Epochs(
            raw,
            events,
            event_id,
            tmin=preprocess_info['tmin'],
            tmax=preprocess_info['tmax'] - 1.0 / raw.info['sfreq'],
            baseline=None,
            preload=True
        )
        data = epochs.get_data()  # (n_epochs, n_channels, n_times)

        for i, (data_idx, label_idx) in enumerate(zip(data, labels)):
            key_bytes = f'Subject{subject_id}_Session{session_num}_Epoch{i}'.encode()
            data_dict = {
                'data': data_idx.astype(np.float32, copy=False),
                'label': int(label_idx) - 1,
                'subject': subject_id,
                'session': session_num,
                'epoch': i
            }
            txn = db.begin(write=True)
            txn.put(key_bytes, pickle.dumps(data_dict))
            txn.commit()
            keys.append(key_bytes)

# Store all keys
txn = db.begin(write=True)
txn.put(b'__keys__', pickle.dumps(keys))
txn.put(b'__meta__', pickle.dumps({"preprocess": preprocess_info}))
txn.commit()
db.close()
