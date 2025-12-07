import os

import torch

from train_common import base_parser, initialize_run, log_dataset_summary, compute_subject_accuracy, finalize, log_confusion_matrix
from util.data_utils import build_cross_subject_loso_loaders
from util.train_utils import build_model, evaluate_subjectwise, train_one_split

def _configure_dbconformer_depth(cfg, dataset):
    if cfg.get("model") != "DBConformer":
        return
    overrides = {
        "BNCI2014001": (2, 2),
        "BNCI2014004": (2, 2),
    }
    tem_depth, chn_depth = overrides.get(dataset, (6, 6))
    model_args = cfg.setdefault("model_args", {})
    model_args["tem_depth"] = tem_depth
    model_args["chn_depth"] = chn_depth

def main():
    parser = base_parser()
    args = parser.parse_args()

    test_method = "cross_subject_loso"
    env = initialize_run(args, test_method)
    cfg = env["cfg"]
    _configure_dbconformer_depth(cfg, env["dataset"])
    device = env["device"]
    log = env["log"]

    n_classes = log_dataset_summary(env, test_method)
    splits = build_cross_subject_loso_loaders(env["lmdb_path"], cfg, dataset_name=env["dataset"])

    per_subject = {}
    fold_results = []
    for split in splits:
        chans, samples = split["chans_samples"]
        n_classes = split["n_classes"]
        model = build_model(env["model_name"], n_classes, chans, samples, env["dataset"], cfg).to(device)
        log("[RESET] model reinitialized for new fold")

        name = f"testS_{split['test_subject']}_valMix"
        tr_n = len(split["train_loader"].dataset)
        va_n = len(split["val_loader"].dataset)
        te_n = len(split["test_loader"].dataset)
        val_detail = ", ".join(
            f"S{sub}:{cnt}" for sub, cnt in sorted(split["val_subject_counts"].items())
        ) or "none"
        log(f"[SPLIT] {name} | train={tr_n} | val={va_n} | test={te_n} | val_mix={val_detail}")

        loaders = {
            "train_loader": split["train_loader"],
            "val_loader": split["val_loader"],
            "test_loader": split["test_loader"],
        }
        best_val = train_one_split(model, device, loaders, cfg, log, split_name=name)

        save_path = os.path.join(env["weights_dir"], f"{name}.pth")
        torch.save(model.state_dict(), save_path)
        log(f"[SAVE ] weights -> {save_path} (best_val_acc={best_val:.4f})")

        per_subj_raw, overall = evaluate_subjectwise(model, device, split["test_loader"])
        subj_acc = compute_subject_accuracy(per_subj_raw)
        test_acc = float(overall["accuracy"])
        test_f1 = float(overall["f1_macro"])
        test_kappa = float(overall["cohen_kappa"])
        fold_results.append({
            "split": name,
            "val_acc": float(best_val),
            "test_acc": test_acc,
            "val_subjects": split["val_subjects"],
            "test_f1_macro": test_f1,
            "test_cohen_kappa": test_kappa,
        })
        log(f"[TEST ] {name} | test_acc={test_acc:.4f} | f1_macro={test_f1:.4f} | kappa={test_kappa:.4f}")
        log_confusion_matrix(env["log_dir"], name, overall["labels"], overall["confusion_matrix"])

        for subj, acc in subj_acc.items():
            per_subject.setdefault(int(subj), []).append(float(acc))

    finalize(env, test_method, n_classes, per_subject, fold_results)


if __name__ == "__main__":
    main()
