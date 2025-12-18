"""
util/log_utils.py
- make_dirs, get_logger, save_results, plot_subject_accuracy
"""

import os, json
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def make_dirs(dataset, model, method):
    tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    wdir = os.path.join("weights", dataset, model, method, tag)
    ldir = os.path.join("log", dataset, model, method, tag)
    os.makedirs(wdir, exist_ok=True); os.makedirs(ldir, exist_ok=True)
    return wdir, ldir

def get_logger(log_dir):
    log_file = os.path.join(log_dir, "train_log.txt")
    def log(msg):
        print(msg, flush=True)
        with open(log_file, "a", encoding="utf-8") as f: f.write(msg + "\n")
    return log

def save_results(log_dir, dataset, model, method, n_classes, per_subject, folds, per_subject_metrics=None):
    path = os.path.join(log_dir, "results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": dataset,
            "model": model,
            "test_method": method,
            "n_classes": n_classes,
            "per_subject_accuracy": per_subject,
            "per_subject_metrics": per_subject_metrics,
            "folds": folds
        }, f, indent=2, ensure_ascii=False)

def plot_subject_accuracy(log_dir, dataset, model, method, n_classes, subject_accs):
    subs = sorted(subject_accs.keys())
    vals = [subject_accs[s] for s in subs]
    mean = float(np.mean(vals)) if vals else 0.0
    std  = float(np.std(vals)) if vals else 0.0

    plt.figure(figsize=(12,5))
    plt.bar([str(s) for s in subs], vals)
    plt.title(f"{dataset} | {model} | {method} | classes={n_classes} | mean={mean:.3f}±{std:.3f}")
    plt.xlabel("Subject")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    out = os.path.join(log_dir, "per_subject_accuracy.png")
    plt.savefig(out, dpi=150); plt.close()
    return out
