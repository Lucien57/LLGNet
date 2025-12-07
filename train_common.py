import argparse, importlib, os, json

import numpy as np
import torch

from util.data_loader import LMDBEEGDataset
from util.log_utils import make_dirs, get_logger, save_results, plot_subject_accuracy

_DROP_RATE_BY_METHOD = {
    "chronological_order": 0.5,
    "within_subject_kfold": 0.5,
    "cross_subject_loso": 0.25,
}

def _override_model_dropouts(cfg, rate):
    if rate is None:
        return

    def apply(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    apply(value)
                elif isinstance(value, (int, float)) and "drop" in key.lower():
                    obj[key] = float(rate)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    apply(item)

    for section in ("model_args", "model"):
        section_cfg = cfg.get(section)
        if isinstance(section_cfg, (dict, list)):
            apply(section_cfg)


def base_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cuda", type=int, default=None, help="CUDA device id (0..7)")
    return parser

def load_config(model_name, seed):
    cfg_mod = importlib.import_module(f"config.config_{model_name}")
    cfg = cfg_mod.model_params
    if seed is not None:
        cfg["train"]["random_seed"] = seed
    return cfg


def get_device(cuda_idx):
    if cuda_idx is not None and torch.cuda.is_available():
        return torch.device(f"cuda:{cuda_idx}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_run_config(log, cfg, dataset):
    def dump_section(name):
        section = cfg.get(name)
        if not section:
            return
        log(f"[CONFIG] {name}:")
        text = json.dumps(section, indent=2, sort_keys=True)
        for line in text.splitlines():
            log(f"[CONFIG]   {line}")

    log("[CONFIG] hyperparameters")
    dump_section("model")
    dump_section("model_args")
    dump_section("train")
    dump_section("data")
    lmdb_paths = cfg.get("paths", {}).get("lmdb_path", {})
    if dataset in lmdb_paths:
        log(f"[CONFIG] paths.lmdb_path.{dataset}={lmdb_paths[dataset]}")


def initialize_run(args, test_method):
    cfg = load_config(args.model, args.seed)
    _override_model_dropouts(cfg, _DROP_RATE_BY_METHOD.get(test_method))
    device = get_device(args.cuda)

    dataset = args.dataset
    if dataset not in cfg["paths"]["lmdb_path"]:
        raise KeyError(f"Dataset {dataset} not found in config['paths']['lmdb_path']")
    lmdb_path = cfg["paths"]["lmdb_path"][dataset]

    weights_dir, log_dir = make_dirs(dataset, args.model, test_method)
    log = get_logger(log_dir)
    log_run_config(log, cfg, dataset)

    return {
        "cfg": cfg,
        "device": device,
        "dataset": dataset,
        "model_name": args.model,
        "lmdb_path": lmdb_path,
        "weights_dir": weights_dir,
        "log_dir": log_dir,
        "log": log,
    }

def log_confusion_matrix(log_dir, split_name, labels, matrix):
    log_file = os.path.join(log_dir, "train_log.txt")
    payload = {
        "split": split_name,
        "labels": list(labels),
        "matrix": matrix.tolist(),
    }
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[CONF ] {payload}\n")


def log_epoch_predictions(log, split_name, epoch, y_true, y_pred):
    payload = {
        "split": split_name,
        "epoch": int(epoch),
        "y_true": [int(v) for v in y_true],
        "y_pred": [int(v) for v in y_pred],
    }
    log(f"[PRED ] {json.dumps(payload)}")


def log_dataset_summary(env, test_method):
    lmdb_path = env["lmdb_path"]
    dataset = env["dataset"]
    model_name = env["model_name"]
    log = env["log"]

    meta_ds = LMDBEEGDataset(lmdb_path)
    n_subjects = int(np.unique(meta_ds.subjects).size)
    n_samples = len(meta_ds)
    chans, samples = meta_ds.chans_samples
    n_classes = meta_ds.n_classes

    log(f"[SETUP] dataset={dataset} | method={test_method} | model={model_name}")
    log(f"[DATA ] subjects={n_subjects} | total_samples={n_samples} | shape=(C,T)=({chans},{samples}) | n_classes={n_classes}")
    preprocess = meta_ds.preprocess_info
    if preprocess:
        band = preprocess.get("bandpass")
        resample = preprocess.get("resample")
        tmin = preprocess.get("tmin")
        tmax = preprocess.get("tmax")
        def fmt_band(bounds):
            if not bounds:
                return "n/a"
            lo, hi = bounds
            if lo is None and hi is None:
                return "n/a"
            if lo is None:
                return f"<{hi:.3g}Hz"
            if hi is None:
                return f">{lo:.3g}Hz"
            return f"[{lo:.3g}, {hi:.3g}]Hz"
        band_str = fmt_band(band)
        resample_str = f"{resample:.3g}Hz" if resample else "n/a"
        window_str = f"{tmin:.3g}s→{tmax:.3g}s" if (tmin is not None and tmax is not None) else "n/a"
        log(f"[DATA ] preprocess bandpass={band_str} | resample={resample_str} | window={window_str}")
    del meta_ds

    return n_classes


def compute_subject_accuracy(per_subject_raw):
    accs = {}
    for subj, payload in per_subject_raw.items():
        y_true = payload["y_true"]
        y_pred = payload["y_pred"]
        if len(y_true) == 0:
            acc = 0.0
        else:
            acc = float(np.mean(y_true == y_pred))
        accs[int(subj)] = acc
    return accs


def finalize(env, test_method, n_classes, per_subject_accs, fold_results):
    log = env["log"]
    dataset = env["dataset"]
    model_name = env["model_name"]

    subject_means = {s: float(np.mean(accs)) for s, accs in per_subject_accs.items()}

    log("\n[RESULT SUMMARY]")
    for subj in sorted(subject_means.keys()):
        msg = f"Subject {int(subj):03d}: {subject_means[subj]:.4f}"
        log(msg)
        print(msg)

    all_means = list(subject_means.values())
    overall_mean = float(np.mean(all_means)) if all_means else 0.0
    overall_std = float(np.std(all_means)) if all_means else 0.0
    log(f"[Overall] mean={overall_mean:.4f} ± {overall_std:.4f}")

    fig = plot_subject_accuracy(env["log_dir"], dataset, model_name, test_method, n_classes, subject_means)
    log(f"[PLOT ] subject accuracy -> {fig}")

    save_results(env["log_dir"], dataset, model_name, test_method, n_classes, subject_means, fold_results)
    log("[DONE ] results.json saved.")
