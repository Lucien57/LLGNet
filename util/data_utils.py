"""
util/data_utils.py
- build_within_subject_kfold_loaders: per-subject k-fold → test=1 fold; from rest, 10% val; rest train
- build_cross_subject_loso_loaders: test=subject i; train=mix of remaining subjects (no dedicated val set)
"""

import pickle, numpy as np, torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from util.data_loader import LMDBEEGDataset
from util.aug import create_augmentation
from util.norm import apply_subject_normalization, resolve_norm_mode

def collate_basic(batch):
    xs = torch.stack([b["x"] for b in batch], 0)  # [B,1,C,T]
    ys = torch.stack([b["y"] for b in batch], 0)  # [B]
    subs = torch.tensor([b["subject"] for b in batch], dtype=torch.long)
    return xs, ys, subs

class _TrainCollate:
    """Callable wrapper so DataLoader workers can pickle the collate_fn on spawn platforms."""
    def __init__(self, augmentation=None):
        self.augmentation = augmentation

    def __call__(self, batch):
        x, y, s = collate_basic(batch)
        if self.augmentation is not None:
            x, y = self.augmentation(x, y)
        return x, y, s


def _make_train_collate(cfg):
    aug_cfg = cfg["data"].get("augmentation")
    aug = create_augmentation(aug_cfg) if (cfg["data"].get("use_augmentation") and aug_cfg) else None
    return _TrainCollate(aug)

def _sort_indices_by_order(indices, trial_orders):
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size <= 1:
        return indices
    order_vals = trial_orders[indices]
    sort_idx = np.argsort(order_vals, kind="stable")
    return indices[sort_idx]

def _group_subject_indices_by_session(ds, subj_indices):
    subj_indices = np.asarray(subj_indices, dtype=np.int64)
    if subj_indices.size == 0:
        return []
    sessions = getattr(ds, "sessions", None)
    if sessions is None or sessions.size == 0:
        return [_sort_indices_by_order(subj_indices, ds.trial_order)]
    subj_sessions = sessions[subj_indices]
    unique_sessions = np.unique(subj_sessions)
    grouped = []
    for sess_id in unique_sessions:
        mask = subj_sessions == sess_id
        sess_idx = subj_indices[mask]
        sess_sorted = _sort_indices_by_order(sess_idx, ds.trial_order)
        if sess_sorted.size == 0:
            continue
        earliest_order = float(ds.trial_order[sess_sorted[0]])
        grouped.append((earliest_order, sess_sorted))
    grouped.sort(key=lambda x: x[0])
    return [g for _, g in grouped if g.size > 0]

def _session_mask_for_dataset(ds, dataset_name):
    if not dataset_name:
        return np.ones(len(ds), dtype=bool)
    # if dataset_name in ("BCIC-IV-2a", "BCIC-IV-2a-2class"):
    #     return np.array([b"SessionT" in key for key in ds.keys], dtype=bool)
    # if dataset_name == "BCIC-IV-2b":
    #     train_sessions = (b"Session01T",)
    #     return np.array([any(sess in key for sess in train_sessions) for key in ds.keys], dtype=bool)
    return np.ones(len(ds), dtype=bool)

_CROSS_SESSION_RULES = {
    "BCIC-IV-2a": {"non_test": ("T",), "test": ("E",)},
    "BCIC-IV-2a-2class": {"non_test": ("T",), "test": ("E",)},
    "BCIC-IV-2b": {"non_test": ("01T", "02T", "03T"), "test": ("04E", "05E")},
}

def _collect_session_labels(ds):
    sess_ids = getattr(ds, "sessions", None)
    if sess_ids is None or len(sess_ids) == 0:
        return {}
    needed = set(int(v) for v in np.unique(sess_ids))
    labels = {}
    if not needed:
        return labels
    with ds.env.begin() as txn:
        for idx, sess_id in enumerate(sess_ids):
            sess_id = int(sess_id)
            if sess_id in labels:
                continue
            blob = txn.get(ds.keys[idx])
            if blob is None:
                continue
            entry = pickle.loads(blob)
            label = entry.get("session")
            if label is None:
                label = "__default_session__"
            else:
                label = str(label)
            labels[sess_id] = label
            if len(labels) == len(needed):
                break
    return labels

def _session_matches(label, patterns):
    if not label or not patterns:
        return False
    norm = str(label).upper()
    for pat in patterns:
        pat_norm = str(pat).upper()
        if not pat_norm:
            continue
        if norm == pat_norm or norm.endswith(pat_norm):
            return True
    return False

def build_within_subject_kfold_loaders(lmdb_path, cfg, val_ratio=0.2, dataset_name=None):
    ds = LMDBEEGDataset(lmdb_path)
    session_mask = _session_mask_for_dataset(ds, dataset_name)
    subs = ds.subjects
    uniq = np.unique(subs[session_mask])
    k = cfg["eval"]["k_folds"]
    splits = []
    norm_mode = resolve_norm_mode(cfg.get("data", {}))

    for subj in uniq:
        idx_subj = np.where(subs == subj)[0]
        idx_subj = idx_subj[session_mask[idx_subj]]
        if idx_subj.size == 0:
            continue
        session_groups = _group_subject_indices_by_session(ds, idx_subj)
        if not session_groups:
            continue
        fold_buckets = [[] for _ in range(k)]
        for group in session_groups:
            per_session_folds = np.array_split(group, k)
            for fold_i, chunk in enumerate(per_session_folds):
                if len(chunk) > 0:
                    fold_buckets[fold_i].append(chunk)
        folds = []
        for fold_i, parts in enumerate(fold_buckets):
            if not parts:
                continue
            te_idx = np.concatenate(parts)
            te_idx = _sort_indices_by_order(te_idx, ds.trial_order)
            folds.append((fold_i, te_idx))
        if not folds:
            continue
        for fold_i, te_idx in folds:
            trva_parts = [part for other_i, part in folds if other_i != fold_i and len(part) > 0]
            if not trva_parts:
                continue
            trva_idx = np.concatenate(trva_parts)
            trva_idx = _sort_indices_by_order(trva_idx, ds.trial_order)
            val_size = max(1, int(len(trva_idx) * val_ratio))
            if val_size >= len(trva_idx):
                val_size = max(1, len(trva_idx) - 1)
            va_idx = trva_idx[-val_size:]
            tr_idx = trva_idx[:-val_size]
            if tr_idx.size == 0:
                continue

            dtr = LMDBEEGDataset(lmdb_path, keys=[ds.keys[j] for j in tr_idx])
            dva = LMDBEEGDataset(lmdb_path, keys=[ds.keys[j] for j in va_idx])
            dte = LMDBEEGDataset(lmdb_path, keys=[ds.keys[j] for j in te_idx])

            # subj_stats, global_stats = _compute_subject_stats(dtr)
            # dtr.transform = SubjectZScoreTransform(subj_stats, global_stats)
            # dva.transform = SubjectZScoreTransform(subj_stats, global_stats)
            # dte.transform = SubjectZScoreTransform(subj_stats, global_stats)
            apply_subject_normalization(norm_mode, dtr, [dtr, dva, dte])

            train_loader = DataLoader(dtr, batch_size=cfg["train"]["batch_size"], shuffle=True,
                                      num_workers=cfg["train"]["num_workers"], collate_fn=_make_train_collate(cfg))
            val_loader   = DataLoader(dva, batch_size=cfg["train"]["batch_size"], shuffle=False,
                                      num_workers=cfg["train"]["num_workers"], collate_fn=collate_basic)
            test_loader  = DataLoader(dte, batch_size=cfg["train"]["batch_size"], shuffle=False,
                                      num_workers=cfg["train"]["num_workers"], collate_fn=collate_basic)

            splits.append({
                "subject_id": int(subj),
                "fold": int(fold_i),
                "train_loader": train_loader,
                "val_loader": val_loader,
                "test_loader": test_loader,
                "n_classes": ds.n_classes,
                "chans_samples": ds.chans_samples
        })
    return splits

def build_cross_session_cs_loaders(lmdb_path, cfg, dataset_name=None, val_ratio=0.2):
    if not dataset_name or dataset_name not in _CROSS_SESSION_RULES:
        raise ValueError(f"Cross-session split not defined for dataset '{dataset_name}'")
    ds = LMDBEEGDataset(lmdb_path)
    session_mask = _session_mask_for_dataset(ds, dataset_name)
    rules = _CROSS_SESSION_RULES[dataset_name]
    session_labels = _collect_session_labels(ds)
    if not session_labels:
        raise ValueError("Cross-session split requires session metadata in LMDB records.")

    session_roles = {}
    for sess_id, label in session_labels.items():
        if _session_matches(label, rules["non_test"]):
            session_roles[sess_id] = "non_test"
        elif _session_matches(label, rules["test"]):
            session_roles[sess_id] = "test"
    missing = [session_labels[sid] for sid in session_labels if sid not in session_roles]
    if missing:
        raise ValueError(f"Unrecognized sessions for dataset '{dataset_name}': {missing}")
    roles_present = set(session_roles.values())
    if "non_test" not in roles_present or "test" not in roles_present:
        raise ValueError(f"Dataset '{dataset_name}' does not provide both non_test and test sessions.")

    subs = ds.subjects
    sess_ids = ds.sessions
    uniq_subs = np.unique(subs[session_mask])
    splits = []
    norm_mode = resolve_norm_mode(cfg.get("data", {}))
    val_ratio = cfg["train"].get("cs_val_ratio", val_ratio)

    for subj in uniq_subs:
        subj_idx = np.where(subs == subj)[0]
        subj_idx = subj_idx[session_mask[subj_idx]]
        if subj_idx.size == 0:
            continue
        non_test_idx = [idx for idx in subj_idx if session_roles.get(int(sess_ids[idx])) == "non_test"]
        test_idx = [idx for idx in subj_idx if session_roles.get(int(sess_ids[idx])) == "test"]
        if len(non_test_idx) < 2 or len(test_idx) == 0:
            continue
        non_test_idx = _sort_indices_by_order(np.asarray(non_test_idx, dtype=np.int64), ds.trial_order)
        test_idx = _sort_indices_by_order(np.asarray(test_idx, dtype=np.int64), ds.trial_order)

        va_size = max(1, int(round(len(non_test_idx) * val_ratio)))
        if va_size >= len(non_test_idx):
            va_size = len(non_test_idx) - 1
        if va_size <= 0:
            continue
        va_idx = non_test_idx[-va_size:]
        tr_idx = non_test_idx[:-va_size]
        if tr_idx.size == 0:
            continue

        dtr = LMDBEEGDataset(lmdb_path, keys=[ds.keys[j] for j in tr_idx])
        dva = LMDBEEGDataset(lmdb_path, keys=[ds.keys[j] for j in va_idx])
        dte = LMDBEEGDataset(lmdb_path, keys=[ds.keys[j] for j in test_idx])

        apply_subject_normalization(norm_mode, dtr, [dtr, dva, dte])

        train_loader = DataLoader(dtr, batch_size=cfg["train"]["batch_size"], shuffle=True,
                                  num_workers=cfg["train"]["num_workers"], collate_fn=_make_train_collate(cfg))
        val_loader   = DataLoader(dva, batch_size=cfg["train"]["batch_size"], shuffle=False,
                                  num_workers=cfg["train"]["num_workers"], collate_fn=collate_basic)
        test_loader  = DataLoader(dte, batch_size=cfg["train"]["batch_size"], shuffle=False,
                                  num_workers=cfg["train"]["num_workers"], collate_fn=collate_basic)

        train_sessions = sorted({session_labels[int(sess_ids[idx])] for idx in np.concatenate([tr_idx, va_idx])})
        test_sessions = sorted({session_labels[int(sess_ids[idx])] for idx in test_idx})

        splits.append({
            "subject_id": int(subj),
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "train_sessions": train_sessions,
            "test_sessions": test_sessions,
            "n_classes": ds.n_classes,
            "chans_samples": ds.chans_samples
        })
    return splits

def build_cross_subject_loso_loaders(lmdb_path, cfg, dataset_name=None):
    ds = LMDBEEGDataset(lmdb_path)
    session_mask = _session_mask_for_dataset(ds, dataset_name)
    subs = ds.subjects
    uniq = np.unique(subs[session_mask])
    splits = []
    norm_mode = resolve_norm_mode(cfg.get("data", {}))
    val_ratio = cfg["train"].get("loso_val_ratio", 0.2)

    for i, test_subj in enumerate(uniq):
        non_test_idx = np.where((subs != test_subj) & session_mask)[0]
        te_idx = np.where((subs == test_subj) & session_mask)[0]
        if te_idx.size == 0 or non_test_idx.size == 0:
            continue

        non_test_labels = ds.labels[non_test_idx]
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio,
                                          random_state=cfg["train"]["random_seed"])
        try:
            tr_rel, va_rel = next(splitter.split(non_test_idx, non_test_labels))
        except ValueError:
            continue
        tr_idx = non_test_idx[tr_rel]
        va_idx = non_test_idx[va_rel]

        dtr = LMDBEEGDataset(lmdb_path, keys=[ds.keys[j] for j in tr_idx])
        dva = LMDBEEGDataset(lmdb_path, keys=[ds.keys[j] for j in va_idx])
        dte = LMDBEEGDataset(lmdb_path, keys=[ds.keys[j] for j in te_idx])

        apply_subject_normalization(norm_mode, dtr, [dtr, dva, dte])

        train_loader = DataLoader(dtr, batch_size=cfg["train"]["batch_size"], shuffle=True,
                                  num_workers=cfg["train"]["num_workers"], collate_fn=_make_train_collate(cfg))
        val_loader   = DataLoader(dva, batch_size=cfg["train"]["batch_size"], shuffle=False,
                                  num_workers=cfg["train"]["num_workers"], collate_fn=collate_basic)
        test_loader  = DataLoader(dte, batch_size=cfg["train"]["batch_size"], shuffle=False,
                                  num_workers=cfg["train"]["num_workers"], collate_fn=collate_basic)

        val_subjects, val_counts = np.unique(subs[va_idx], return_counts=True)
        splits.append({
            "test_subject": int(test_subj),
            "val_subjects": [int(s) for s in val_subjects],
            "val_subject_counts": {int(s): int(c) for s, c in zip(val_subjects, val_counts)},
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "n_classes": ds.n_classes,
            "chans_samples": ds.chans_samples
        })
    return splits

def build_chronological_split_loaders(lmdb_path, cfg, ratio=0.8, dataset_name=None):
    ds = LMDBEEGDataset(lmdb_path)
    session_mask = _session_mask_for_dataset(ds, dataset_name)
    subs = ds.subjects
    uniq = np.unique(subs[session_mask])
    splits = []
    norm_mode = resolve_norm_mode(cfg.get("data", {}))

    for subj in uniq:
        idx_subj = np.where(subs == subj)[0]
        idx_subj = idx_subj[session_mask[idx_subj]]
        if idx_subj.size == 0:
            continue
        session_groups = _group_subject_indices_by_session(ds, idx_subj)
        if not session_groups:
            continue
        tr_parts, te_parts = [], []
        for group in session_groups:
            split_point = int(len(group) * ratio)
            tr_parts.append(group[:split_point])
            te_parts.append(group[split_point:])
        tr_parts = [part for part in tr_parts if len(part) > 0]
        te_parts = [part for part in te_parts if len(part) > 0]
        tr_idx = np.concatenate(tr_parts) if tr_parts else np.array([], dtype=np.int64)
        te_idx = np.concatenate(te_parts) if te_parts else np.array([], dtype=np.int64)
        tr_idx = _sort_indices_by_order(tr_idx, ds.trial_order) if tr_idx.size else tr_idx
        te_idx = _sort_indices_by_order(te_idx, ds.trial_order) if te_idx.size else te_idx

        dtr = LMDBEEGDataset(lmdb_path, keys=[ds.keys[j] for j in tr_idx])
        dva = None  # no val set in CO
        dte = LMDBEEGDataset(lmdb_path, keys=[ds.keys[j] for j in te_idx])

        # subj_stats, global_stats = _compute_subject_stats(dtr)
        # dtr.transform = SubjectZScoreTransform(subj_stats, global_stats)
        # dte.transform = SubjectZScoreTransform(subj_stats, global_stats)
        apply_subject_normalization(norm_mode, dtr, [dtr, dte])

        train_loader = DataLoader(dtr, batch_size=cfg["train"]["batch_size"], shuffle=True,
                                  num_workers=cfg["train"]["num_workers"], collate_fn=_make_train_collate(cfg))
        test_loader  = DataLoader(dte, batch_size=cfg["train"]["batch_size"], shuffle=False,
                                  num_workers=cfg["train"]["num_workers"], collate_fn=collate_basic)

        splits.append({
            "subject_id": int(subj),
            "fold": 0,
            "train_loader": train_loader,
            "test_loader": test_loader,
            "n_classes": ds.n_classes,
            "chans_samples": ds.chans_samples
        })
    return splits


import copy
import scipy.signal as signal
class filterBank(object):  # From FBCNet and FBMSNet
    
    """
    filter the given signal in the specific bands using cheby2 iir filtering.
    If only one filter is specified then it acts as a simple filter and returns 2d matrix
    Else, the output will be 3d with the filtered signals appended in the third dimension.
    axis is the time dimension along which the filtering will be applied
    """

    def __init__(self, filtBank, fs, filtAllowance=2, axis=1, filtType='filter'):
        self.filtBank = filtBank
        self.fs = fs
        self.filtAllowance =filtAllowance
        self.axis = axis
        self.filtType=filtType

    def bandpassFilter(self, data, bandFiltCutF,  fs, filtAllowance=2, axis=1, filtType='filter'):
        """
         Filter a signal using cheby2 iir filtering.

        Parameters
        ----------
        data: 2d/ 3d np array
            trial x channels x time
        bandFiltCutF: two element list containing the low and high cut off frequency in hertz.
            if any value is specified as None then only one sided filtering will be performed
        fs: sampling frequency
        filtAllowance: transition bandwidth in hertz
        filtType: string, available options are 'filtfilt' and 'filter'

        Returns
        -------
        dataOut: 2d/ 3d np array after filtering
            Data after applying bandpass filter.
        """
        aStop = 30 # stopband attenuation
        aPass = 3 # passband attenuation
        nFreq= fs/2 # Nyquist frequency
        
        if (bandFiltCutF[0] == 0 or bandFiltCutF[0] is None) and (bandFiltCutF[1] == None or bandFiltCutF[1] >= fs / 2.0):
            # no filter
            print("Not doing any filtering. Invalid cut-off specifications")
            return data
        
        elif bandFiltCutF[0] == 0 or bandFiltCutF[0] is None:
            # low-pass filter
            print("Using lowpass filter since low cut hz is 0 or None")
            fPass =  bandFiltCutF[1]/ nFreq
            fStop =  (bandFiltCutF[1]+filtAllowance)/ nFreq
            # find the order
            [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
            b, a = signal.cheby2(N, aStop, fStop, 'lowpass')
        
        elif (bandFiltCutF[1] is None) or (bandFiltCutF[1] == fs / 2.0):
            # high-pass filter
            print("Using highpass filter since high cut hz is None or nyquist freq")
            fPass =  bandFiltCutF[0]/ nFreq
            fStop =  (bandFiltCutF[0]-filtAllowance)/ nFreq
            # find the order
            [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
            b, a = signal.cheby2(N, aStop, fStop, 'highpass')
        
        else:
            # band-pass filter
            # print("Using bandpass filter")
            fPass =  (np.array(bandFiltCutF)/ nFreq).tolist()
            fStop =  [(bandFiltCutF[0]-filtAllowance)/ nFreq, (bandFiltCutF[1]+filtAllowance)/ nFreq]
            # find the order
            [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
            b, a = signal.cheby2(N, aStop, fStop, 'bandpass')

        if filtType == 'filtfilt':
            dataOut = signal.filtfilt(b, a, data, axis=axis)
        else:
            dataOut = signal.lfilter(b, a, data, axis=axis)
        return dataOut

    def __call__(self, data1):

        data = copy.deepcopy(data1)
        d = data['data']

        # initialize output
        out = np.zeros([*d.shape, len(self.filtBank)])  # [N, C, T, n_bands]
        # print(out.shape)
        
        # repetitively filter the data.
        for i, filtBand in enumerate(self.filtBank):
            out[:,:,:,i] = self.bandpassFilter(d, filtBand, self.fs, self.filtAllowance,
                    self.axis, self.filtType)

        # remove any redundant 3rd dimension
        if len(self.filtBank) <= 1:
            out =np.squeeze(out, axis = -1)

        data['data'] = torch.from_numpy(out).float()
        return data
