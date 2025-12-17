#!/usr/bin/env python3
"""Run Careful/Modified/CTSNet trial configs (recap/pooling grid).

This script mirrors the scheduling pattern from ``run_all_experiments.py``:
  - Distributes jobs across available GPUs (one job per GPU at a time).
  - Each job is a call to ``train_loso.py`` (LOSO) or ``train_co.py`` (CO)
    with ``-m`` pointing at a trial config (e.g., ``CTSNet1``).
  - After each run, the newly created log directory under ``log/`` is
    mirrored into ``log_config_trials/``.
  - When all runs finish, it prints a table with configs on rows and
    test methods on columns (LOSO, CO), using mean per-subject accuracy
    from ``results.json``.

Usage example:
    python try_multiple_config.py \
        --datasets BCIC-IV-2a BCIC-IV-2b \
        --configs CTSNet1 CTSNet2 \
        --methods loso cs

Defaults (editable near the top of this file):
  datasets: BCIC-IV-2a, BCIC-IV-2b
  configs : all configs found under config_trial/config_*.py
  methods : loso, cs
  GPUs    : prefer 4, 5, 6, 7 (override via --devices)
"""

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


# Map short method names to entry scripts and labels
METHOD_ALIAS = {
    "loso": "train_loso.py",
    "cs": "train_cs.py",
    "co": "train_co.py",
}

TEST_METHOD_NAMES = {
    "train_loso.py": "cross_subject_loso",
    "train_cs.py": "cross_session",
    "train_co.py": "chronological_order",
}

METHOD_LABELS = {
    "train_loso.py": "LOSO",
    "train_cs.py": "CS",
    "train_co.py": "CO",
}

SUMMARY_METHOD_ORDER = ("train_loso.py", "train_cs.py", "train_co.py")

LOG_ROOT = "log"
LOG_MIRROR_ROOT = "log_config_trials"

def discover_trial_models(config_dir: str = "config_trial") -> Sequence[str]:
    """Return model names for every config_* file in config_trial."""

    if not os.path.isdir(config_dir):
        return ()

    models: List[str] = []
    prefix = "config_"
    suffix = ".py"
    for entry in sorted(os.listdir(config_dir)):
        if not entry.startswith(prefix) or not entry.endswith(suffix):
            continue
        models.append(entry[len(prefix) : -len(suffix)])
    return tuple(models)


TRIAL_MODELS: Sequence[str] = discover_trial_models()

EEGNET_GRID_MODELS: Sequence[str] = (
    "EEGNet_plain",
    "EEGNet_two_stage",
    "EEGNet_adv",
    "EEGNet_adv_two_stage",
)

# Editable defaults (configs, methods, datasets, GPUs)
DEFAULT_TRIAL_MODELS: Sequence[str] = tuple(
    model for model in EEGNET_GRID_MODELS if model in TRIAL_MODELS
) or TRIAL_MODELS
DEFAULT_METHODS: Sequence[str] = ("loso", "cs")
DEFAULT_DATASETS: Sequence[str] = ("BCIC-IV-2a", "BCIC-IV-2b")
PREFERRED_GPUS: Sequence[int] = (6, 7)


def list_existing_runs(base_dir: str) -> Set[str]:
    if not os.path.isdir(base_dir):
        return set()
    return {
        entry
        for entry in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, entry))
    }


def mirror_new_runs(dataset: str, model: str, method_script: str, new_entries: Set[str]) -> None:
    if not new_entries:
        return
    test_dir = TEST_METHOD_NAMES[method_script]
    for entry in sorted(new_entries):
        src = os.path.join(LOG_ROOT, dataset, model, test_dir, entry)
        dst = os.path.join(LOG_MIRROR_ROOT, dataset, model, test_dir, entry)
        if not os.path.isdir(src):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"[MIRROR] {dataset}/{model}/{test_dir}/{entry} -> {LOG_MIRROR_ROOT}", flush=True)


def extract_accuracy_stats(log_dir: str, entries: Set[str]) -> Optional[Tuple[float, float]]:
    if not entries:
        return None

    latest_entry = None
    results_path = None
    for entry in sorted(entries, reverse=True):
        candidate = os.path.join(log_dir, entry, "results.json")
        if os.path.isfile(candidate):
            latest_entry = entry
            results_path = candidate
            break
        print(f"[WARN] Missing results.json for {candidate}", flush=True)

    if not results_path:
        return None

    try:
        with open(results_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        print(f"[WARN] Could not parse {results_path}: {exc}", flush=True)
        return None

    per_subject = data.get("per_subject_accuracy") or {}
    values = [float(v) for v in per_subject.values() if v is not None]
    if not values:
        print(f"[WARN] No per-subject accuracies found in {results_path}", flush=True)
        return None

    acc = mean(values)
    std = pstdev(values) if len(values) > 1 else 0.0
    print(
        f"[STATS ] {os.path.dirname(results_path)} | mean_acc={acc:.4f} ± {std:.4f}",
        flush=True,
    )
    return acc, std


def formatted_name(method_script: str, model: str, dataset: str, seed: Optional[int]) -> str:
    stem = method_script.rsplit(".", 1)[0]
    seed_suffix = "" if seed is None else f" | seed={seed}"
    return f"{stem} | {model} | {dataset}{seed_suffix}"


def worker(
    gpu_label: int,
    task_queue: "queue.Queue[Tuple[str, str, str, Optional[int]]]",
    results: List[Tuple[str, int, int]],
    lock: threading.Lock,
    summary: Dict[str, Dict[str, Dict[str, List[Tuple[Optional[int], float, float]]]]],
) -> None:
    while True:
        try:
            method_script, model, dataset, seed = task_queue.get_nowait()
        except queue.Empty:
            break

        label = formatted_name(method_script, model, dataset, seed)
        print(f"[GPU {gpu_label}] START {label}", flush=True)

        env = os.environ.copy()
        # Bind this worker to a single physical GPU; training code uses --cuda 0
        # so the model runs on the first device in CUDA_VISIBLE_DEVICES.
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_label)

        test_dir = TEST_METHOD_NAMES[method_script]
        log_dir = os.path.join(LOG_ROOT, dataset, model, test_dir)
        before_logs = list_existing_runs(log_dir)

        cmd = ["python", method_script, "-m", model, "-d", dataset, "--cuda", "0"]
        if seed is not None:
            cmd += ["--seed", str(seed)]

        try:
            completed = subprocess.run(cmd, env=env, check=False)
            rc = int(completed.returncode)
        except Exception as exc:  # pragma: no cover - defensive logging
            rc = -1
            print(f"[GPU {gpu_label}] ERROR {label}: {exc}", flush=True)
        finally:
            task_queue.task_done()

        status = "OK" if rc == 0 else f"FAIL (rc={rc})"
        print(f"[GPU {gpu_label}] {status} {label}", flush=True)

        after_logs = list_existing_runs(log_dir)
        new_entries = after_logs - before_logs
        mirror_new_runs(dataset, model, method_script, new_entries)
        stats = extract_accuracy_stats(log_dir, new_entries)

        with lock:
            results.append((label, gpu_label, rc))
            if stats is not None:
                method_bucket = summary.setdefault(dataset, {}).setdefault(model, {}).setdefault(
                    method_script, []
                )
                acc, std = stats
                method_bucket.append((seed, acc, std))


def _format_method_stats(entries: Optional[Iterable[Tuple[Optional[int], float, float]]]) -> str:
    if not entries:
        return "N/A"

    parts = []
    for seed, acc, std in entries:
        seed_prefix = "seed=default" if seed is None else f"seed={seed}"
        parts.append(f"{seed_prefix}: {acc:.4f} ± {std:.4f}")
    return "; ".join(parts)


def print_config_summary(
    dataset: str,
    models: Sequence[str],
    methods_used: Sequence[str],
    summary: Dict[str, Dict[str, List[Tuple[Optional[int], float, float]]]],
) -> str:
    print(f"\n=== Trial Config Summary for {dataset} ===")

    method_order = [m for m in SUMMARY_METHOD_ORDER if m in methods_used]
    headers = ["Config"] + [METHOD_LABELS[m] for m in method_order]

    # Build rows: config identifiers on Y axis
    rows: List[List[str]] = []
    for model in models:
        # Expect names like "Conformer_tEA" or "EEGNet_tZscore" -> "EA"/"Zscore"
        suffix = model.split("_t")[-1]
        cfg_label = suffix
        row = [cfg_label]
        model_stats = summary.get(model, {})
        for m in method_order:
            stats = model_stats.get(m)
            cell = _format_method_stats(stats)
            row.append(cell)
        rows.append(row)

    # Column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    header_line = " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    sep_line = "-+-".join("-" * widths[i] for i in range(len(headers)))

    print(header_line)
    print(sep_line)
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))

    # Return the table as a string so we can save it
    lines = [header_line, sep_line]
    lines.extend(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) for row in rows)
    return "\n".join(lines) + "\n"


def detect_gpus() -> List[int]:
    """Detect available GPUs via torch, fallback to [0] if unknown.

    Using [0] in a CPU-only environment is harmless because the training
    code falls back to CPU when CUDA is not available.
    """

    try:
        import torch

        count = torch.cuda.device_count()
        if count <= 0:
            return [0]
        return list(range(count))
    except Exception:
        return [0]


def choose_gpus(devices: Optional[Sequence[int]]) -> List[int]:
    """Prefer CUDA devices 5, 6, and 7 when available, otherwise fall back.

    Users can still override the list explicitly via --devices.
    """

    if devices:
        return list(devices)

    detected = detect_gpus()
    allowed = [gpu for gpu in PREFERRED_GPUS if gpu in detected]
    if allowed:
        return allowed

    print(
        f"No preferred GPUs {PREFERRED_GPUS} detected; aborting. Use --devices to override if needed.",
        flush=True,
    )
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        "-d",
        nargs="*",
        default=list(DEFAULT_DATASETS),
        help="Datasets to run (e.g. BCIC-IV-2a BCIC-IV-2b)",
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=list(DEFAULT_TRIAL_MODELS),
        help="Subset of trial model names under config_trial/ (e.g. AdEEGNet1)",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=list(DEFAULT_METHODS),
        choices=list(METHOD_ALIAS.keys()),
        help="Which evaluation methods to run: loso, cs, co",
    )
    parser.add_argument("--seed", type=int, default=None, help="Single random seed override")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Try multiple seeds in one run (e.g. --seeds 1 2 3)",
    )
    parser.add_argument(
        "--devices",
        nargs="*",
        type=int,
        default=None,
        help="CUDA device ids to use (default: preferred GPUs or detected list)",
    )

    args = parser.parse_args()

    if args.seed is not None and args.seeds:
        parser.error("Use either --seed or --seeds, not both.")

    seeds: List[Optional[int]]
    if args.seeds:
        # Preserve order, keep unique seeds
        seen = set()
        seeds = []
        for s in args.seeds:
            if s not in seen:
                seen.add(s)
                seeds.append(int(s))
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [None]

    if not TRIAL_MODELS:
        print("No trial configs found under config_trial/config_*.py.")
        sys.exit(1)

    invalid_models = [m for m in args.configs if m not in TRIAL_MODELS]
    if invalid_models:
        print("Ignoring unknown config names:")
        print("  " + " ".join(invalid_models))

    models = [m for m in args.configs if m in TRIAL_MODELS]
    if not models:
        print("No valid trial models selected. Available:")
        print("  " + " ".join(TRIAL_MODELS))
        sys.exit(1)

    methods_used = [METHOD_ALIAS[m] for m in args.methods]

    # Determine GPUs to use
    gpus = choose_gpus(args.devices)
    print(f"Using GPUs: {gpus}")

    os.makedirs(LOG_MIRROR_ROOT, exist_ok=True)

    # Prepare task queue: each job is (method_script, model, dataset, seed)
    tasks: "queue.Queue[Tuple[str, str, str, Optional[int]]]" = queue.Queue()
    for dataset in args.datasets:
        for method_script in methods_used:
            for model in models:
                for seed in seeds:
                    tasks.put((method_script, model, dataset, seed))

    results: List[Tuple[str, int, int]] = []
    lock = threading.Lock()
    # summary[dataset][model][method_script] = List[(seed, acc, std)]
    summary: Dict[str, Dict[str, Dict[str, List[Tuple[Optional[int], float, float]]]]] = {}
    combined_tables: List[str] = []

    threads: List[threading.Thread] = [
        threading.Thread(
            target=worker,
            args=(gpu, tasks, results, lock, summary),
            daemon=True,
        )
        for gpu in gpus
    ]

    for t in threads:
        t.start()

    tasks.join()
    for t in threads:
        t.join()

    failures = [item for item in results if item[2] != 0]

    print("\n=== Trial Job Summary ===")
    for label, gpu, rc in sorted(results):
        outcome = "OK" if rc == 0 else f"FAIL (rc={rc})"
        print(f"{outcome:10s} | GPU {gpu} | {label}")

    for dataset in args.datasets:
        dataset_summary = summary.get(dataset, {})
        table_text = print_config_summary(dataset, models, methods_used, dataset_summary)
        combined_tables.append(f"=== Trial Config Summary for {dataset} ===\n{table_text}")

        # Save summary table under log_config_trials for later inspection
        summary_dir = os.path.join(LOG_MIRROR_ROOT, dataset)
        os.makedirs(summary_dir, exist_ok=True)
        summary_path = os.path.join(summary_dir, "config_trial_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as handle:
            handle.write(table_text)
        print(f"\nSummary table saved to {summary_path}")

    if combined_tables:
        combined_path = "config_trial_summary.txt"
        with open(combined_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(combined_tables))
        print(f"\nCombined summary table saved to {combined_path}")

    if failures:
        print("\nSome trial runs failed. Check logs above and under log/.")
        sys.exit(1)

    print("\nAll trial runs completed.")


if __name__ == "__main__":
    main()
