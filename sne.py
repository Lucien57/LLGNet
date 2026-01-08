#!/usr/bin/env python3
"""
new_val_fixed.py

Safe t-SNE / feature extraction for plain vs adv encoders.
Handles checkpoint shape mismatch and flexible extractor names.
Usage example:
python new_val_fixed.py \
  --model_plain DeepConvNet_plain --weights_plain /path/to/plain.pth \
  --model_adv DeepConvNet_adv   --weights_adv   /path/to/adv.pth \
  --dataset BCIC-IV-2a --cuda 0 --batch_size 64 --n_vis 2000
"""
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from train_common import load_config, get_device
from util.data_loader import LMDBEEGDataset
from util.data_utils import collate_basic
from util.train_utils import build_model
from sklearn.metrics import silhouette_score
# ---------------- utils ----------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_extractor(module: nn.Module) -> nn.Module:
    """Return a submodule to use as feature extractor. Works with many model APIs."""
    if hasattr(module, "backbone"):
        return module.backbone
    if hasattr(module, "get_embedding"):
        return module  # has method get_embedding; extractor will be called differently
    for name in ("encoder", "enc", "convnet"):
        if hasattr(module, name):
            return getattr(module, name)
    # last resort, return module itself
    return module

def safe_load_matched(model: nn.Module, ckpt_path: str):
    """Load checkpoint but only matching keys/shapes; report ignored keys."""
    state = torch.load(ckpt_path, map_location="cpu")
    ms = model.state_dict()
    matched = {k: v for k, v in state.items() if k in ms and v.shape == ms[k].shape}
    ignored = [k for k in state.keys() if k not in matched]
    if ignored:
        print(f"Ignored keys due to mismatch: {ignored}")
    model.load_state_dict(matched, strict=False)
    return matched

@torch.no_grad()
def extract_features_from_extractor(extractor: nn.Module, loader: DataLoader, device: torch.device):
    """
    extractor: a module (or model with get_embedding)
    loader: yields (x, y_task, y_subj)  (or similar)
    returns: feats (N, D) numpy array, subj_ids (N,) numpy array, y_tasks (N,) numpy array
    """
    extractor = extractor.to(device)
    extractor.eval()
    feats_list = []
    subj_list = []
    task_list = []
    for batch in loader:
        # handle (x, y_task, y_subj) or (x, y_subj)
        if len(batch) == 3:
            xb, y_task, y_subj = batch
        elif len(batch) == 2:
            xb, y_subj = batch
            # no explicit task label: create dummy -1
            y_task = torch.full((xb.size(0),), -1, dtype=torch.long)
        else:
            xb = batch[0]
            y_subj = batch[-1]
            # try to extract task label as second item if exists
            if len(batch) >= 3:
                y_task = batch[1]
            else:
                y_task = torch.full((xb.size(0),), -1, dtype=torch.long)

        xb = xb.to(device)
        if hasattr(extractor, "get_embedding"):
            out = extractor.get_embedding(xb)
        else:
            out = extractor(xb)
        if out.dim() > 2:
            out = out.view(out.size(0), -1)
        feats_list.append(out.cpu().numpy())

        # y_subj, y_task may be torch tensors or numpy arrays
        if isinstance(y_subj, torch.Tensor):
            subj_list.append(y_subj.cpu().numpy())
        else:
            subj_list.append(np.asarray(y_subj))
        if isinstance(y_task, torch.Tensor):
            task_list.append(y_task.cpu().numpy())
        else:
            task_list.append(np.asarray(y_task))

    feats = np.concatenate(feats_list, axis=0) if feats_list else np.zeros((0,0))
    subs = np.concatenate(subj_list, axis=0) if subj_list else np.zeros((0,), dtype=int)
    tasks = np.concatenate(task_list, axis=0) if task_list else np.zeros((0,), dtype=int)
    return feats, subs, tasks


def visualize_tsne(features, subjects, title="t-SNE", perplexity=30, n_iter=1000, save_path=None):
    print(f"Running t-SNE on {features.shape[0]} samples (dim={features.shape[1]})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, init='pca', random_state=42)
    emb = tsne.fit_transform(features)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=emb[:,0], y=emb[:,1], hue=subjects, palette="tab10", s=30, alpha=0.8, linewidth=0)
    plt.title(title)
    plt.legend(title='Subject', bbox_to_anchor=(1.02,1), loc='upper left')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200)
        print(f"[SAVE] t-SNE figure saved to {save_path}")
    plt.show()
    indices = np.random.choice(len(features), min(2000, len(features)), replace=False)
    X_sub = features[indices]
    y_sub = subjects[indices]
    score = silhouette_score(X_sub, y_sub)
    print(f"Silhouette Score (Subject Clustering): {score:.4f}")
    return emb


# ---------------- main ----------------
def main(args):
    
    set_seed(args.seed)
    device = get_device(args.cuda)

    # load configs (we use plain config to find dataset path; if they differ, user can adapt)
    cfg_plain = load_config(args.model_plain, args.seed)
    cfg_adv   = load_config(args.model_adv, args.seed)

    lmdb_path = cfg_plain["paths"]["lmdb_path"].get(args.dataset)
    if lmdb_path is None:
        raise KeyError(f"Dataset {args.dataset} not found in config paths.")
    full_ds = LMDBEEGDataset(lmdb_path)

    test_loader = DataLoader(full_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_basic)

    chans, samples = full_ds.chans_samples
    n_classes = int(full_ds.n_classes)

    # build models separately
    model_plain = build_model(args.model_plain, n_classes, chans, samples, args.dataset, cfg_plain).to(device)
    model_adv   = build_model(args.model_adv,   n_classes, chans, samples, args.dataset, cfg_adv).to(device)

    # safe load ckpts (only matching keys)
    safe_load_matched(model_plain, args.weights_plain)
    safe_load_matched(model_adv,   args.weights_adv)
    print("[LOAD] models loaded.")

    # get extractors (handles many model APIs)
    ex_plain = get_extractor(model_plain)
    ex_adv   = get_extractor(model_adv)
    # freeze params (not necessary but safe)
    for p in ex_plain.parameters():
        p.requires_grad = False
    for p in ex_adv.parameters():
        p.requires_grad = False

    # extract features
    feat_plain, subs, tasks = extract_features_from_extractor(ex_plain, test_loader, device)
    feat_adv, subs2, tasks2 = extract_features_from_extractor(ex_adv, test_loader, device)
    if not np.array_equal(subs, subs2):
        print("Warning: subject arrays differ between extractors. Using first extractor's subjects.")
    if not np.array_equal(tasks, tasks2):
        print("Warning: task label arrays differ between extractors. Using first extractor's tasks.")
    tasks = tasks  # use tasks from first extractor


        # parse trial labels requested (args.trial_labels is a list of ints or None)
    if args.trial_labels is not None and len(args.trial_labels) > 0:
        # find indices where tasks is in requested set
        mask = np.isin(tasks, np.array(args.trial_labels, dtype=int))
        if mask.sum() == 0:
            raise RuntimeError(f"No samples found for trial_labels={args.trial_labels}")
        # filter both feature arrays & subject array
        feat_plain = feat_plain[mask]
        feat_adv   = feat_adv[mask]
        subs       = subs[mask]
        print(f"[FILTER] kept {feat_plain.shape[0]} samples for trial_labels={args.trial_labels}")
    else:
        print(f"[FILTER] keeping all {feat_plain.shape[0]} samples (no trial_labels specified)")

    # sample indices (same for both)
    N_VIS = min(args.n_vis, feat_plain.shape[0])
    rng = np.random.RandomState(args.seed)
    idx = rng.choice(feat_plain.shape[0], N_VIS, replace=False)
    feat_plain_vis = feat_plain[idx]
    feat_adv_vis   = feat_adv[idx]
    subs_vis       = subs[idx]

    # optional standardization (helps t-SNE stability)
    mu = feat_plain_vis.mean(0, keepdims=True)
    sd = feat_plain_vis.std(0, keepdims=True) + 1e-10
    feat_plain_vis = (feat_plain_vis - mu) / sd
    feat_adv_vis   = (feat_adv_vis - mu) / sd

    # visualize
    emb_plain = visualize_tsne(
    feat_plain_vis, subs_vis, 
    title="Plain encoder t-SNE", 
    perplexity=args.perplexity, 
    n_iter=args.tsne_iter,
    save_path="tsne_plain_type0.png"   # <-- 保存图片
)
    emb_adv = visualize_tsne(
    feat_adv_vis, subs_vis, 
    title="Adversarial encoder t-SNE", 
    perplexity=args.perplexity, 
    n_iter=args.tsne_iter,
    save_path="tsne_adv_type0.png"     # <-- 保存图片
)

    # save embeddings optionally
    if args.save_emb:
        np.save("emb_plain.npy", emb_plain)
        np.save("emb_adv.npy", emb_adv)
        np.save("subs.npy", subs_vis)
        print("[SAVE] saved emb_plain.npy, emb_adv.npy, subs.npy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_plain", required=True)
    parser.add_argument("--weights_plain", required=True)
    parser.add_argument("--model_adv", required=True)
    parser.add_argument("--weights_adv", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cuda", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_vis", type=int, default=2000)
    parser.add_argument("--perplexity", type=int, default=30)
    parser.add_argument("--tsne_iter", type=int, default=1000)
    parser.add_argument("--save_emb", action="store_true")
    parser.add_argument("--trial_labels", type=str, default=None,
                    help="Comma-separated list of task/trial labels to visualize (e.g. '0,1'). If omitted, use all trials.")

    args = parser.parse_args()
    if args.trial_labels is None:
        args.trial_labels = None
    else:
        args.trial_labels = [int(x) for x in args.trial_labels.split(",") if x.strip()!='']

    main(args)
