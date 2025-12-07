"""
data_loader.py
- LMDBEEGDataset with proper @property accessors.
- Sample format: x -> [1, C, T] float32; y -> long; subject -> int.
"""

import lmdb, pickle, numpy as np, torch
from torch.utils.data import Dataset

class LMDBEEGDataset(Dataset):
    def __init__(self, lmdb_path, keys=None, transform=None):
        super().__init__()
        self._lmdb_path = lmdb_path
        self._lmdb_env_args = dict(readonly=True, lock=False, readahead=True, meminit=False)
        self.env = self._open_env()
        with self.env.begin() as txn:
            all_keys = pickle.loads(txn.get(b"__keys__"))
            meta_blob = txn.get(b"__meta__")
        self.keys = keys if keys is not None else all_keys
        self.transform = transform
        self._meta = pickle.loads(meta_blob) if meta_blob else {}

        labels, subjects, shapes = [], [], set()
        orders = []
        sessions = []
        session_lookup = {}
        abs_max = 0.0
        rms_accum = 0.0
        rms_count = 0
        with self.env.begin() as txn:
            for k in self.keys:
                d = pickle.loads(txn.get(k))
                labels.append(int(d["label"]))
                subjects.append(int(d["subject"]))
                shapes.add(tuple(d["data"].shape))
                order_val = d.get("trial_order")
                if order_val is None:
                    order_val = len(orders)
                orders.append(float(order_val))
                sess_key = d.get("session")
                if sess_key is None:
                    sess_key = "__default_session__"
                else:
                    sess_key = str(sess_key)
                sess_id = session_lookup.setdefault(sess_key, len(session_lookup))
                sessions.append(sess_id)
                # track magnitude to decide scaling heuristic later
                curr_max = float(np.abs(d["data"]).max())
                if curr_max > abs_max:
                    abs_max = curr_max
                rms_accum += float(np.mean(np.square(d["data"], dtype=np.float64)))
                rms_count += 1
        self._labels = np.asarray(labels, dtype=np.int64)
        self._subjects = np.asarray(subjects, dtype=np.int64)
        self._shape = next(iter(shapes))
        self._n_classes = int(self._labels.max() + 1) if len(self._labels)>0 else 0
        self._trial_order = np.asarray(orders, dtype=np.float32) if orders else np.arange(len(self.keys), dtype=np.float32)
        if sessions:
            self._sessions = np.asarray(sessions, dtype=np.int64)
        else:
            self._sessions = np.zeros(len(self.keys), dtype=np.int64)

        meta_scale = None if not self._meta else self._meta.get("scale_to_uv")
        if meta_scale is not None:
            self._scale_to_uv = float(meta_scale)
        else:
            rms = float(np.sqrt(rms_accum / max(1, rms_count))) if rms_count else 0.0
            # heuristic: RMS <1e-4 -> volts (x1e6), <1e-1 -> millivolts (x1e3), otherwise already microvolts
            if rms < 1e-4:
                self._scale_to_uv = 1e6
            elif rms < 1e-1:
                self._scale_to_uv = 1e3
            else:
                self._scale_to_uv = 1.0
        self._abs_max_est = abs_max
        self._rms_est = float(np.sqrt(rms_accum / max(1, rms_count))) if rms_count else 0.0
        if self._meta is not None:
            self._meta.setdefault("scale_to_uv", self._scale_to_uv)

    @property
    def scale_to_uv(self):
        return self._scale_to_uv

    @property
    def labels(self): return self._labels

    @property
    def subjects(self): return self._subjects

    @property
    def n_classes(self): return self._n_classes

    @property
    def chans_samples(self): return self._shape  # (C, T)

    @property
    def preprocess_info(self):
        return self._meta.get("preprocess") if self._meta else None

    @property
    def trial_order(self):
        return self._trial_order

    @property
    def sessions(self):
        return self._sessions

    def __len__(self): return len(self.keys)

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            d = pickle.loads(txn.get(self.keys[idx]))
        # data = d["data"].astype(np.float32, copy=False)  # original scale
        data = d["data"].astype(np.float32, copy=False)
        order = float(d.get("trial_order", idx))
        if getattr(self, "_scale_to_uv", 1.0) != 1.0:
            data = data * np.float32(self._scale_to_uv)
        x = np.expand_dims(data, 0)  # [1,C,T]
        y = int(d["label"])
        s = int(d["subject"])
        sample = {
            "x": torch.from_numpy(x),
            "y": torch.tensor(y, dtype=torch.long),
            "subject": s,
            "order": torch.tensor(order, dtype=torch.float32),
        }
        if self.transform: sample = self.transform(sample)
        return sample

    def _open_env(self):
        return lmdb.open(self._lmdb_path, **self._lmdb_env_args)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["env"] = None  # LMDB environments are not picklable
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.env = self._open_env()
