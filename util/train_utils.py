"""
util/train_utils.py 
- build_model, train_without_val, train_one_split, evaluate_subjectwise
"""

import torch, torch.nn as nn, numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, cohen_kappa_score
from torch.optim import Adam, AdamW
from torch.utils.data import ConcatDataset, DataLoader

def _unpack_loader_batch(batch):
    if not isinstance(batch, (list, tuple)):
        raise TypeError(f"Unexpected batch type: {type(batch)}")
    if len(batch) == 3:
        xb, yb, sb = batch
        sess = None
    elif len(batch) == 4:
        xb, yb, sb, sess = batch
    else:
        raise ValueError(f"Unexpected batch length: {len(batch)}")
    return xb, yb, sb, sess

def _merge_train_val_loaders(train_loader, val_loader):
    if train_loader is None or val_loader is None:
        return train_loader
    merged_ds = ConcatDataset([train_loader.dataset, val_loader.dataset])
    batch_size = getattr(train_loader, "batch_size", 1)
    num_workers = getattr(train_loader, "num_workers", 0)
    drop_last = getattr(train_loader, "drop_last", False)
    pin_memory = getattr(train_loader, "pin_memory", False)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "drop_last": drop_last,
        "pin_memory": pin_memory,
    }
    if getattr(train_loader, "persistent_workers", False) and num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    collate_fn = getattr(train_loader, "collate_fn", None)
    if collate_fn is not None:
        loader_kwargs["collate_fn"] = collate_fn
    return DataLoader(merged_ds, **loader_kwargs)

def build_model(model_name, n_classes, chans, samples, dataset_name, cfg):
    model_key = cfg.get("model", model_name)
    model_args = cfg.get("model_args")
    if model_args is None and isinstance(cfg.get("model"), dict):
        model_args = cfg.get("model")
    base_args = dict(model_args or {})

    def _pop(mapping, *names, default=None):
        for name in names:
            if name in mapping:
                return mapping.pop(name)
        return default

    def _assign(args_dict, params_dict, name, *aliases, default=None):
        val = _pop(args_dict, name, *aliases, default=default)
        if val is not None:
            params_dict[name] = val
        return val

    class _MLPClassifier(nn.Module):
        def __init__(self, in_dim, out_dim, hidden=None, dropout=0.0):
            super().__init__()
            layers = []
            last = in_dim
            if hidden:
                layers.extend([nn.Linear(last, hidden), nn.GELU(), nn.Dropout(dropout)])
                last = hidden
            layers.append(nn.Linear(last, out_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            return self.net(x)

    if model_key == "Conformer":
        from models.Conformer import Conformer
        args = dict(base_args)
        emb_size = _pop(args, "emb_size", "embed_dim")
        if emb_size is not None:
            args["emb_size"] = emb_size
        params = {"Chans": chans, "n_classes": n_classes}
        params.update(args)
        print(params)
        return Conformer(**params)

    elif model_key == "EEGNet":
        from models.EEGNet import EEGNet, AdversarialEEGNet
        args = dict(base_args)
        use_adv = bool(_pop(args, "enable_adversarial_head", "use_adversarial_head", default=False))
        adv_lambda = _pop(args, "adv_lambda", default=0.01)
        n_nuisance = _pop(args, "n_nuisance", "n_nuis", default=None)
        params = {"n_classes": n_classes, "Chans": chans, "Samples": samples}
        _assign(args, params, "kernLength")
        _assign(args, params, "F1")
        _assign(args, params, "D")
        _assign(args, params, "F2")
        _assign(args, params, "dropoutRate")
        _assign(args, params, "norm_rate")
        if not use_adv:
            return EEGNet(**params)
        adv_kwargs = dict(params)
        adv_kwargs.pop("n_classes", None)
        return AdversarialEEGNet(
            n_classes=params["n_classes"],
            n_nuisance=n_nuisance or chans,
            lambd=adv_lambda,
            **adv_kwargs,
        )
        
    elif model_key == "DeepConvNet":
        from models.DeepConvNet import DeepConvNet
        args = dict(base_args)
        params = {"n_classes": n_classes, "Chans": chans, "Samples": samples}
        _assign(args, params, "dropoutRate")
        _assign(args, params, "batch_norm")
        _assign(args, params, "batch_norm_alpha")
        return DeepConvNet(**params)
    
    elif model_key == "ShallowConvNet":
        from models.ShallowConvNet import ShallowConvNet
        args = dict(base_args)
        params = {"n_classes": n_classes, "Chans": chans, "Samples": samples}
        _assign(args, params, "dropoutRate")
        _assign(args, params, "batch_norm")
        _assign(args, params, "batch_norm_alpha")
        return ShallowConvNet(**params)

def train_without_val(model, device, train_loader, test_loader, cfg, log, split_name):
    betas = cfg["train"].get("betas", (0.9, 0.999))
    opt_name = cfg["train"].get("optimizer", "Adam")
    if opt_name == "Adam":     
        opt = Adam(model.parameters(), lr=cfg["train"]["learning_rate"],
               weight_decay=cfg["train"].get("weight_decay", 0.0), betas=betas)
    elif opt_name == "AdamW":
        opt = AdamW(model.parameters(), lr=cfg["train"]["learning_rate"],
               weight_decay=cfg["train"].get("weight_decay", 0.0), betas=betas)
    else:
        pass
    crit = nn.CrossEntropyLoss()
    adv_lambd = cfg["train"].get("adv_lambda", 0.0)##adv
    max_ep = cfg["train"]["max_epochs"]
    best_acc = -1.0
    best_loss = float("inf")
    best_state = None

    for ep in range(1, max_ep + 1):
        model.train()
        losses, correct, total = [], 0, 0
        adv_correct, adv_total = 0, 0
        for batch in train_loader:
            xb, yb, sb, _ = _unpack_loader_batch(batch)
            xb, yb, sb = xb.to(device), yb.to(device), sb.to(device)
            opt.zero_grad()
            logits = model(xb)
            if isinstance(logits, tuple):
                cla_logits, adv_logits = logits
            else:
                cla_logits, adv_logits = logits, None

            loss = crit(cla_logits, yb)
            if adv_logits is not None and adv_lambd > 0:
                loss = loss + adv_lambd * crit(adv_logits, sb)
            loss.backward()
            opt.step()

            losses.append(loss.item())
            preds = torch.argmax(cla_logits, 1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
            if adv_logits is not None:
                adv_preds = torch.argmax(adv_logits, 1)
                adv_correct += (adv_preds == sb).sum().item()
                adv_total += sb.size(0)

        tr_loss = float(np.mean(losses)) if losses else 0.0
        tr_acc = (correct / total) if total else 0.0
        tr_adv_acc = (adv_correct / adv_total) if adv_total else None
        if tr_acc > best_acc or (tr_acc == best_acc and tr_loss < best_loss):
            best_acc = tr_acc
            best_loss = tr_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        te_loss, te_acc, te_adv_acc = float("nan"), float("nan"), None
        if test_loader is not None:
            model.eval()
            te_losses, te_correct, te_total = [], 0, 0
            adv_correct, adv_total = 0, 0
            with torch.no_grad():
                for batch in test_loader:
                    xb, yb, sb, _ = _unpack_loader_batch(batch)
                    xb, yb, sb = xb.to(device), yb.to(device), sb.to(device)
                    logits = model(xb)
                    if isinstance(logits, tuple):
                        cla_logits, adv_logits = logits
                    else:
                        cla_logits, adv_logits = logits, None
                    loss = crit(cla_logits, yb)
                    te_losses.append(loss.item())
                    preds = torch.argmax(cla_logits, 1)
                    te_correct += (preds == yb).sum().item()
                    te_total += yb.size(0)
                    if adv_logits is not None:
                        adv_preds = torch.argmax(adv_logits, 1)
                        adv_correct += (adv_preds == sb).sum().item()
                        adv_total += sb.size(0)
            te_loss = float(np.mean(te_losses)) if te_losses else 0.0
            te_acc = (te_correct / te_total) if te_total else 0.0
            te_adv_acc = (adv_correct / adv_total) if adv_total else None

        msg = f"[Epoch {ep:03d}] train_loss={tr_loss:.4f} cla_acc={tr_acc:.4f}"
        if tr_adv_acc is not None:
            msg += f" adv_acc={tr_adv_acc:.4f}"
        msg += f" || test_loss={te_loss:.4f} cla_acc={te_acc:.4f}"
        if te_adv_acc is not None:
            msg += f" adv_acc={te_adv_acc:.4f}"
        log(msg)

    if best_state is not None:
        model.load_state_dict(best_state)
    return float(best_acc)

def train_one_split(model, device, loaders, cfg, log, split_name=None):
    betas = cfg["train"].get("betas", (0.9, 0.999))
    adv_lambd = cfg["train"].get("adv_lambda", 0.0)
    opt_name = cfg["train"].get("optimizer", "Adam")

    def _make_optimizer():
        if opt_name == "Adam":
            return Adam(model.parameters(), lr=cfg["train"]["learning_rate"],
                        weight_decay=cfg["train"].get("weight_decay", 0.0), betas=betas)
        if opt_name == "AdamW":
            return AdamW(model.parameters(), lr=cfg["train"]["learning_rate"],
                         weight_decay=cfg["train"].get("weight_decay", 0.0), betas=betas)
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    opt = _make_optimizer()
    crit = nn.CrossEntropyLoss()

    stage1_epochs = cfg["train"]["max_epochs"]
    stage2_epochs = int(cfg["train"].get("two_stage_extra_epochs", 30))
    two_stage = bool(cfg["train"].get("two_stage", False))
    early_stop_ep = stage1_epochs
    patience = cfg["train"]["early_stopping_patience"]
    train_loader = loaders["train_loader"]
    val_loader = loaders.get("val_loader")
    has_val = val_loader is not None
    best_state, best_metric, best_loss, no_imp = None, -1.0, float("inf"), 0
    best_test = 0.0
    test_loader = loaders.get("test_loader")
    stage2_ran = False

    for ep in range(1, stage1_epochs + 1):
        model.train()
        tr_loss = []
        yt, yp = [], []
        adv_correct, adv_total = 0, 0
        for batch in train_loader:
            xb, yb, sb, _ = _unpack_loader_batch(batch)
            xb, yb, sb = xb.to(device), yb.to(device), sb.to(device)
            opt.zero_grad()
            logits = model(xb)
            if isinstance(logits, tuple):
                cla_logits, adv_logits = logits
            else:
                cla_logits, adv_logits = logits, None

            loss = crit(cla_logits, yb)
            if adv_logits is not None and adv_lambd > 0:
                loss = loss + adv_lambd * crit(adv_logits, sb)
            loss.backward()
            opt.step()

            tr_loss.append(loss.item())
            yt.extend(yb.cpu().numpy())
            yp.extend(torch.argmax(cla_logits, 1).cpu().numpy())
            if adv_logits is not None:
                adv_preds = torch.argmax(adv_logits, 1)
                adv_correct += (adv_preds == sb).sum().item()
                adv_total += sb.size(0)
        tr_acc = accuracy_score(yt, yp) if yt else 0.0
        tr_loss = float(np.mean(tr_loss)) if tr_loss else 0.0
        tr_adv_acc = (adv_correct / adv_total) if adv_total else None

        if has_val:
            vl_loss_vals = []
            yv, pv = [], []
            adv_val_correct, adv_val_total = 0, 0
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    xb, yb, sb, _ = _unpack_loader_batch(batch)
                    xb, yb, sb = xb.to(device), yb.to(device), sb.to(device)
                    logits = model(xb)
                    if isinstance(logits, tuple):
                        cla_logits, adv_logits = logits
                    else:
                        cla_logits, adv_logits = logits, None
                    loss = crit(cla_logits, yb)
                    if adv_logits is not None and adv_lambd > 0:
                        loss = loss + adv_lambd * crit(adv_logits, sb)
                    vl_loss_vals.append(loss.item())
                    yv.extend(yb.cpu().numpy())
                    pv.extend(torch.argmax(cla_logits, 1).cpu().numpy())
                    if adv_logits is not None:
                        adv_preds = torch.argmax(adv_logits, 1)
                        adv_val_correct += (adv_preds == sb).sum().item()
                        adv_val_total += sb.size(0)
            vl_acc = accuracy_score(yv, pv) if yv else 0.0
            vl_loss = float(np.mean(vl_loss_vals)) if vl_loss_vals else 0.0
            vl_adv_acc = (adv_val_correct / adv_val_total) if adv_val_total else None
        else:
            vl_acc = None
            vl_loss = None
            vl_adv_acc = None

        test_acc = None
        test_adv_acc = None
        if test_loader is not None:
            te_true, te_pred = [], []
            adv_test_correct, adv_test_total = 0, 0
            with torch.no_grad():
                for batch in test_loader:
                    xb, yb, sb, _ = _unpack_loader_batch(batch)
                    xb, sb = xb.to(device), sb.to(device)
                    logits = model(xb)
                    if isinstance(logits, tuple):
                        cla_logits, adv_logits = logits
                    else:
                        cla_logits, adv_logits = logits, None
                    te_true.extend(yb.cpu().numpy())
                    te_pred.extend(torch.argmax(cla_logits, 1).cpu().numpy())
                    if adv_logits is not None:
                        adv_preds = torch.argmax(adv_logits, 1)
                        adv_test_correct += (adv_preds == sb).sum().item()
                        adv_test_total += sb.size(0)
            test_acc = accuracy_score(te_true, te_pred) if te_true else 0.0
            test_adv_acc = (adv_test_correct / adv_test_total) if adv_test_total else None

        prefix = f"[Epoch {ep:03d}]"
        msg = f"{prefix} train_loss={tr_loss:.4f} cla_acc={tr_acc:.4f}"
        if tr_adv_acc is not None:
            msg += f" adv_acc={tr_adv_acc:.4f}"
        if has_val:
            msg += f" || val_loss={vl_loss:.4f} cla_acc={vl_acc:.4f}"
            if vl_adv_acc is not None:
                msg += f" adv_acc={vl_adv_acc:.4f}"
        if test_acc is not None:
            msg += f" || test_acc={test_acc:.4f}"
            if test_adv_acc is not None:
                msg += f" adv_acc={test_adv_acc:.4f}"
            msg += f" || best_test={best_test:.4f}"
        log(msg)

        if test_acc is not None and test_acc > best_test:
            best_test = test_acc

        metric = vl_acc if has_val else tr_acc
        loss_for_metric = vl_loss if has_val else tr_loss
        if metric is not None and (metric > best_metric or (metric == best_metric and loss_for_metric < best_loss)):
            best_metric = metric
            best_loss = loss_for_metric
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        elif has_val:
            no_imp += 1
            if no_imp >= patience and ep > early_stop_ep:
                log(f"[EarlyStop] epoch={ep} best_val_acc={best_metric:.4f}")
                break

    if two_stage:
        if not has_val:
            log("[TwoStage] Requested but no validation loader is available; skipping stage 2.")
        elif stage2_epochs <= 0:
            log("[TwoStage] Extra epoch count <= 0; skipping stage 2.")
        else:
            merged_loader = _merge_train_val_loaders(train_loader, val_loader)
            merged_count = len(merged_loader.dataset) if hasattr(merged_loader, "dataset") else (len(train_loader.dataset) + len(val_loader.dataset))
            log(f"[TwoStage] Stage 1 complete (best_val_acc={best_metric:.4f}). "
                f"Merging train({len(train_loader.dataset)}) + val({len(val_loader.dataset)}) -> "
                f"{merged_count} samples and running {stage2_epochs} more epochs.")
            if best_state is not None:
                model.load_state_dict(best_state)
            opt = _make_optimizer()
            stage2_ran = True
            for extra_idx in range(1, stage2_epochs + 1):
                ep = stage1_epochs + extra_idx
                model.train()
                tr_loss = []
                yt, yp = [], []
                adv_correct, adv_total = 0, 0
                for batch in merged_loader:
                    xb, yb, sb, _ = _unpack_loader_batch(batch)
                    xb, yb, sb = xb.to(device), yb.to(device), sb.to(device)
                    opt.zero_grad()
                    logits = model(xb)
                    if isinstance(logits, tuple):
                        cla_logits, adv_logits = logits
                    else:
                        cla_logits, adv_logits = logits, None
                    loss = crit(cla_logits, yb)
                    if adv_logits is not None and adv_lambd > 0:
                        loss = loss + adv_lambd * crit(adv_logits, sb)
                    loss.backward()
                    opt.step()
                    tr_loss.append(loss.item())
                    yt.extend(yb.cpu().numpy())
                    yp.extend(torch.argmax(cla_logits, 1).cpu().numpy())
                    if adv_logits is not None:
                        adv_preds = torch.argmax(adv_logits, 1)
                        adv_correct += (adv_preds == sb).sum().item()
                        adv_total += sb.size(0)
                tr_acc = accuracy_score(yt, yp) if yt else 0.0
                tr_loss_val = float(np.mean(tr_loss)) if tr_loss else 0.0
                tr_adv_acc = (adv_correct / adv_total) if adv_total else None

                test_acc = None
                test_adv_acc = None
                if test_loader is not None:
                    te_true, te_pred = [], []
                    adv_test_correct, adv_test_total = 0, 0
                    with torch.no_grad():
                        for batch in test_loader:
                            xb, yb, sb, _ = _unpack_loader_batch(batch)
                            xb, sb = xb.to(device), sb.to(device)
                            logits = model(xb)
                            if isinstance(logits, tuple):
                                cla_logits, adv_logits = logits
                            else:
                                cla_logits, adv_logits = logits, None
                            te_true.extend(yb.cpu().numpy())
                            te_pred.extend(torch.argmax(cla_logits, 1).cpu().numpy())
                            if adv_logits is not None:
                                adv_preds = torch.argmax(adv_logits, 1)
                                adv_test_correct += (adv_preds == sb).sum().item()
                                adv_test_total += sb.size(0)
                    test_acc = accuracy_score(te_true, te_pred) if te_true else 0.0
                    test_adv_acc = (adv_test_correct / adv_test_total) if adv_test_total else None

                prefix = f"[Epoch {ep:03d}][S2]"
                msg = f"{prefix} train_loss={tr_loss_val:.4f} cla_acc={tr_acc:.4f}"
                if tr_adv_acc is not None:
                    msg += f" adv_acc={tr_adv_acc:.4f}"
                if test_acc is not None:
                    msg += f" || test_acc={test_acc:.4f}"
                    if test_adv_acc is not None:
                        msg += f" adv_acc={test_adv_acc:.4f}"
                    msg += f" || best_test={best_test:.4f}"
                log(msg)

                if test_acc is not None and test_acc > best_test:
                    best_test = test_acc

    if not stage2_ran and best_state is not None:
        model.load_state_dict(best_state)
    return best_metric

def evaluate_subjectwise(model, device, loader):
    from collections import defaultdict
    model.eval(); y_true_all = []; y_pred_all = []; subjects_all = []
    with torch.no_grad():
        for batch in loader:
            xb, yb, sb, _ = _unpack_loader_batch(batch)
            xb = xb.to(device)
            logits = model(xb)
            if isinstance(logits, tuple):
                cla_logits, _ = logits
            else:
                cla_logits = logits
            pred = torch.argmax(cla_logits, 1).cpu().numpy()
            y_true_all.extend(yb.numpy()); y_pred_all.extend(pred); subjects_all.extend(sb.numpy())

    labels = list(range(loader.dataset.n_classes))
    subj_targets = defaultdict(list); subj_preds = defaultdict(list)
    for t, p, s in zip(y_true_all, y_pred_all, subjects_all):
        subj_targets[int(s)].append(t)
        subj_preds[int(s)].append(p)

    per_subject = {}
    for subj in sorted(subj_targets):
        y_true = np.asarray(subj_targets[subj])
        y_pred = np.asarray(subj_preds[subj])
        per_subject[int(subj)] = {
            "y_true": y_true,
            "y_pred": y_pred,
        }

    if y_true_all:
        acc = float(accuracy_score(y_true_all, y_pred_all))
        f1_macro = float(f1_score(y_true_all, y_pred_all, labels=labels, average="macro", zero_division=0))
        kappa = float(cohen_kappa_score(y_true_all, y_pred_all, labels=labels))
        conf = confusion_matrix(y_true_all, y_pred_all, labels=labels)
    else:
        acc, f1_macro, kappa = 0.0, 0.0, 0.0
        conf = np.zeros((len(labels), len(labels)), dtype=int)

    overall = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "cohen_kappa": kappa,
        "confusion_matrix": conf,
        "labels": labels,
    }
    return per_subject, overall

