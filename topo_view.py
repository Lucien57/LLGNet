#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import torch.nn as nn

from sklearn.linear_model import LogisticRegression

from torch.utils.data import DataLoader, Subset

from train_common import load_config, get_device
from util.data_loader import LMDBEEGDataset
from util.data_utils import collate_basic
from util.train_utils import build_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -------------------------
def get_extractor(model):
    if hasattr(model, "backbone"):
        return model.backbone
    if hasattr(model, "get_embedding"):
        return model
    for n in ("encoder", "enc", "convnet"):
        if hasattr(model, n):
            return getattr(model, n)
    return model


@torch.no_grad()
def extract_features(extractor, loader, device):
    extractor.eval()
    feats, subjects = [], []
    for xb, _, sb in loader:
        xb = xb.to(device)
        out = extractor(xb)
        if out.dim() > 2:
            out = out.view(out.size(0), -1)
        feats.append(out.cpu().numpy())
        subjects.append(sb.numpy())
    return np.concatenate(feats), np.concatenate(subjects)


def load_model(name, weights, cfg, chans, samples, n_classes, dataset, device):
    model = build_model(name, n_classes, chans, samples, dataset, cfg).to(device)
    ckpt = torch.load(weights, map_location="cpu")
    ms = model.state_dict()
    filtered = {k: v for k, v in ckpt.items()
                if k in ms and v.shape == ms[k].shape}
    model.load_state_dict(filtered, strict=False)
    return model


# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_plain", required=True)
    parser.add_argument("--weights_plain", required=True)
    parser.add_argument("--model_adv", required=True)
    parser.add_argument("--weights_adv", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--cuda", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = load_config(args.model_plain, args.seed)
    device = get_device(args.cuda)

    lmdb_path = cfg["paths"]["lmdb_path"][args.dataset]
    ds = LMDBEEGDataset(lmdb_path)

    chans, samples = ds.chans_samples
    n_classes = int(ds.n_classes)
    subjects_all = np.array(ds.subjects)
    unique_subjects = np.unique(subjects_all)

    model_plain = load_model(
        args.model_plain, args.weights_plain,
        cfg, chans, samples, n_classes,
        args.dataset, device
    )
    model_adv = load_model(
        args.model_adv, args.weights_adv,
        cfg, chans, samples, n_classes,
        args.dataset, device
    )

    enc_plain = get_extractor(model_plain).to(device)
    enc_adv = get_extractor(model_adv).to(device)

    for p in enc_plain.parameters():
        p.requires_grad = False
    for p in enc_adv.parameters():
        p.requires_grad = False

    conf_plain, conf_adv = [], []

    # -------------------------
    # LOSO loop
    # -------------------------
    for test_subj in unique_subjects:
        train_idx = np.where(subjects_all != test_subj)[0]
        test_idx  = np.where(subjects_all == test_subj)[0]

        train_loader = DataLoader(
            Subset(ds, train_idx),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_basic
        )
        test_loader = DataLoader(
            Subset(ds, test_idx),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_basic
        )

        Xtr_p, ytr = extract_features(enc_plain, train_loader, device)
        Xte_p, _   = extract_features(enc_plain, test_loader, device)

        Xtr_a, _ = extract_features(enc_adv, train_loader, device)
        Xte_a, _ = extract_features(enc_adv, test_loader, device)

        # remap subjects in train
        uniq = np.unique(ytr)
        mp = {s: i for i, s in enumerate(uniq)}
        ytr_mc = np.array([mp[s] for s in ytr])

        clf_p = LogisticRegression(max_iter=2000, multi_class="multinomial")
        clf_a = LogisticRegression(max_iter=2000, multi_class="multinomial")

        clf_p.fit(Xtr_p, ytr_mc)
        clf_a.fit(Xtr_a, ytr_mc)

        # confidence on unseen subject
        conf_p = clf_p.predict_proba(Xte_p).max(axis=1).mean()
        conf_a = clf_a.predict_proba(Xte_a).max(axis=1).mean()

        conf_plain.append(conf_p)
        conf_adv.append(conf_a)

        print(f"Subject {int(test_subj):2d}: plain_conf={conf_p:.4f}, adv_conf={conf_a:.4f}")

    conf_plain = np.array(conf_plain)
    conf_adv = np.array(conf_adv)

    print("=" * 60)
    print(f"Mean confidence plain: {conf_plain.mean():.4f}")
    print(f"Mean confidence adv  : {conf_adv.mean():.4f}")
    print("=" * 60)
    # 提取所有样本的特征和 subject 标签
    full_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_basic)
    features_plain, subjects_all = extract_features(enc_plain, full_loader, device)
    features_adv, _ = extract_features(enc_adv, full_loader, device)
 

    # --- 1. 划分训练/测试集 ---
    X_train_p, X_test_p, y_train, y_test = train_test_split(
        features_plain, subjects_all, test_size=0.2, stratify=subjects_all, random_state=42
    )
    X_train_a, X_test_a, _, _ = train_test_split(
        features_adv, subjects_all, test_size=0.2, stratify=subjects_all, random_state=42
    )

    print(f"Probe Training on {len(y_train)} samples, Testing on {len(y_test)} samples")
    print(f"Class count (Subject IDs): {len(np.unique(subjects_all))}")
    chance_level = 1.0 / len(np.unique(subjects_all))
    print(f"Random Chance Level: {chance_level:.4f}")

    # --- 3. 训练 Probe (Plain) ---
    print('Training Probe on Plain Encoder...')
    clf_p = LogisticRegression(max_iter=1000, multi_class='multinomial')
    clf_p.fit(X_train_p, y_train)
    acc_p = accuracy_score(y_test, clf_p.predict(X_test_p))

    # --- 4. 训练 Probe (Adv) ---
    print('Training Probe on Adversarial Encoder...')
    clf_a = LogisticRegression(max_iter=1000, multi_class='multinomial')
    clf_a.fit(X_train_a, y_train)
    acc_a = accuracy_score(y_test, clf_a.predict(X_test_a))

    print("="*30)
    print(f"Plain Probe Accuracy: {acc_p:.4f} (Should be HIGH)")
    print(f"Adv   Probe Accuracy: {acc_a:.4f} (Should be near {chance_level:.4f})")
    print("="*30)
if __name__ == "__main__":
    main()
