"""
util/norm.py
- Subject-level normalization utilities shared by data loaders.
"""

from collections import defaultdict
import numpy as np
import torch


class SubjectZScoreTransform:
    def __init__(self, subject_stats, global_stats, eps=1e-6):
        self.subject_stats = {int(k): (v[0].clone(), v[1].clone()) for k, v in subject_stats.items()}
        self.global_stats = (global_stats[0].clone(), global_stats[1].clone()) if global_stats else None
        self.eps = eps

    def __call__(self, sample):
        subj = int(sample["subject"])
        x = sample["x"].float()
        mean_std = self.subject_stats.get(subj)
        if mean_std is None:
            if self.global_stats is not None:
                mean_std = (self.global_stats[0], self.global_stats[1])
            else:
                mean = x.mean(dim=0, keepdim=True)
                std = torch.clamp(x.std(dim=0, keepdim=True), min=self.eps)
                mean_std = (mean, std)
            self.subject_stats[subj] = (mean_std[0].clone(), mean_std[1].clone())
        mean, std = mean_std
        std = std.clamp_min(self.eps)
        normed = (x - mean) / std
        sample = dict(sample)
        sample["x"] = normed.to(sample["x"].dtype)
        return sample


class SubjectEAStatic:
    """Applies precomputed EA matrices on the training set."""

    def __init__(self, subject_mats):
        self.subject_mats = {
            int(k): torch.as_tensor(v, dtype=torch.float32).clone()
            for k, v in (subject_mats or {}).items()
        }

    def __call__(self, sample):
        subj = int(sample["subject"])
        mat = self.subject_mats.get(subj)
        if mat is None:
            return sample
        x = sample["x"].float().squeeze(0)
        aligned = torch.matmul(mat, x)
        sample = dict(sample)
        sample["x"] = aligned.unsqueeze(0).to(sample["x"].dtype)
        return sample


class SubjectEASequentialPrecomputed:
    """Sequential EA that reuses cumulative covariances for validation/test splits."""

    def __init__(self, dataset, eps=1e-6):
        self.lookup = {}
        self.eps = eps
        self._prepare(dataset)

    def _prepare(self, dataset):
        by_subj = defaultdict(list)
        orders = getattr(dataset, "trial_order", None)
        for idx in range(len(dataset)):
            subj = int(dataset.subjects[idx])
            order_val = float(orders[idx]) if orders is not None else float(idx)
            order_key = int(round(order_val))
            by_subj[subj].append((order_key, idx))
        for subj, seq in by_subj.items():
            seq.sort(key=lambda t: (t[0], t[1]))
            ref = None
            count = 0
            for order_key, idx in seq:
                sample = dataset[idx]
                x = sample["x"].float().squeeze(0).numpy()
                cov = np.cov(x) + self.eps * np.eye(x.shape[0], dtype=np.float32)
                if ref is None:
                    ref = cov
                else:
                    ref = (ref * count + cov) / (count + 1)
                count += 1
                mat = _matrix_power_neg_half(ref, self.eps)
                self.lookup[(int(subj), order_key)] = torch.from_numpy(mat)

    def __call__(self, sample):
        subj = int(sample["subject"])
        order_tensor = sample.get("order")
        if isinstance(order_tensor, torch.Tensor):
            order_key = int(round(float(order_tensor.item())))
        else:
            order_key = int(round(float(order_tensor)))
        mat = self.lookup.get((subj, order_key))
        if mat is None:
            return sample
        x = sample["x"].float().squeeze(0)
        aligned = torch.matmul(mat, x)
        sample = dict(sample)
        sample["x"] = aligned.unsqueeze(0).to(sample["x"].dtype)
        return sample


def _compute_subject_stats(dataset, eps=1e-6):
    if len(dataset) == 0:
        return {}, None
    sums, sumsq, counts = {}, {}, {}
    global_sum = None
    global_sumsq = None
    total = 0
    for idx in range(len(dataset)):
        sample = dataset[idx]
        x = sample["x"].float()
        subj = int(sample["subject"])
        if subj not in sums:
            sums[subj] = torch.zeros_like(x)
            sumsq[subj] = torch.zeros_like(x)
            counts[subj] = 0
        sums[subj] += x
        sumsq[subj] += x * x
        counts[subj] += 1
        if global_sum is None:
            global_sum = torch.zeros_like(x)
            global_sumsq = torch.zeros_like(x)
        global_sum += x
        global_sumsq += x * x
        total += 1
    stats = {}
    for subj, total_sum in sums.items():
        mean = total_sum / counts[subj]
        var = sumsq[subj] / counts[subj] - mean * mean
        std = torch.sqrt(torch.clamp(var, min=0.0)).clamp_min(eps)
        stats[int(subj)] = (mean, std)
    if total == 0:
        return stats, None
    global_mean = global_sum / total
    global_var = global_sumsq / total - global_mean * global_mean
    global_std = torch.sqrt(torch.clamp(global_var, min=0.0)).clamp_min(eps)
    return stats, (global_mean, global_std)


def _matrix_power_neg_half(matrix, eps=1e-6):
    vals, vecs = np.linalg.eigh(matrix)
    vals = np.clip(vals, eps, None).astype(np.float64, copy=False)
    inv_sqrt = np.diag(vals ** -0.5)
    result = vecs @ inv_sqrt @ vecs.T
    return result.astype(np.float32)


def _compute_subject_ea(dataset, eps=1e-6):
    if dataset is None or len(dataset) == 0:
        return {}
    cov_sums, counts = {}, {}
    for idx in range(len(dataset)):
        sample = dataset[idx]
        x = sample["x"].float().squeeze(0).numpy()
        subj = int(sample["subject"])
        cov = np.cov(x) + eps * np.eye(x.shape[0], dtype=np.float32)
        if subj not in cov_sums:
            cov_sums[subj] = cov
            counts[subj] = 1
        else:
            cov_sums[subj] += cov
            counts[subj] += 1
    subject_mats = {
        int(subj): _matrix_power_neg_half(cov_sums[subj] / counts[subj], eps)
        for subj in cov_sums
    }
    return subject_mats


def resolve_norm_mode(data_cfg):
    norm = data_cfg.get("Norm")
    if norm is None:
        if data_cfg.get("use_zscore_normalization", False):
            return "zscore"
        if data_cfg.get("use_ea_normalization", False):
            return "ea"
        return "none"
    if isinstance(norm, str):
        norm_lower = norm.strip().lower()
        if norm_lower in ("zscore", "z-score", "z"):
            return "zscore"
        if norm_lower in ("ea", "euclideanalignment", "euclidean_alignment"):
            return "ea"
        if norm_lower in ("none", "null", "off"):
            return "none"
    raise ValueError(f"Unsupported normalization mode: {norm}")


def apply_subject_normalization(norm_mode, train_dataset, datasets, eps=1e-6):
    if norm_mode == "none":
        return
    if norm_mode == "zscore":
        subj_stats, global_stats = _compute_subject_stats(train_dataset, eps=eps)
        transforms = [SubjectZScoreTransform(subj_stats, global_stats) if ds is not None else None for ds in datasets]
    elif norm_mode == "ea":
        subject_mats = _compute_subject_ea(train_dataset, eps=eps)
        transforms = []
        for ds in datasets:
            if ds is None:
                transforms.append(None)
            elif ds is train_dataset:
                transforms.append(SubjectEAStatic(subject_mats))
            else:
                transforms.append(SubjectEASequentialPrecomputed(ds, eps=eps))
    else:
        raise ValueError(f"Unsupported normalization mode: {norm_mode}")
    for ds, transform in zip(datasets, transforms):
        if ds is not None:
            ds.transform = transform
