"""
util/train_utils.py 
- build_model, train_without_val, train_one_split, evaluate_subjectwise
"""

import torch, torch.nn as nn, numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, cohen_kappa_score
from torch.optim import Adam, AdamW

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
        from models.EEGNet import EEGNet
        args = dict(base_args)
        params = {"n_classes": n_classes, "Chans": chans, "Samples": samples}
        _assign(args, params, "kernLength")
        _assign(args, params, "F1")
        _assign(args, params, "D")
        _assign(args, params, "F2")
        _assign(args, params, "dropoutRate")
        _assign(args, params, "norm_rate")
        return EEGNet(**params)
        
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
    max_ep = cfg["train"]["max_epochs"]
    best_acc = -1.0
    best_loss = float("inf")
    best_state = None

    for ep in range(1, max_ep + 1):
        model.train()
        losses, correct, total = [], 0, 0
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

            losses.append(loss.item())
            preds = torch.argmax(logits, 1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        tr_loss = float(np.mean(losses)) if losses else 0.0
        tr_acc = (correct / total) if total else 0.0
        if tr_acc > best_acc or (tr_acc == best_acc and tr_loss < best_loss):
            best_acc = tr_acc
            best_loss = tr_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        te_loss, te_acc = float("nan"), float("nan")
        if test_loader is not None:
            model.eval()
            te_losses, te_correct, te_total = [], 0, 0
            with torch.no_grad():
                for xb, yb, _ in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    loss = crit(logits, yb)
                    te_losses.append(loss.item())
                    preds = torch.argmax(logits, 1)
                    te_correct += (preds == yb).sum().item()
                    te_total += yb.size(0)
            te_loss = float(np.mean(te_losses)) if te_losses else 0.0
            te_acc = (te_correct / te_total) if te_total else 0.0

        log(f"[Epoch {ep:03d}] train_loss={tr_loss:.4f} acc={tr_acc:.4f} || test_loss={te_loss:.4f} acc={te_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return float(best_acc)

def train_one_split(model, device, loaders, cfg, log, split_name=None):
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
    
    max_ep = cfg["train"]["max_epochs"]
    early_stop_ep = max_ep
    patience = cfg["train"]["early_stopping_patience"]
    val_loader = loaders.get("val_loader")
    has_val = val_loader is not None
    best_state, best_metric, best_loss, no_imp = None, -1.0, float("inf"), 0
    best_test = 0
    test_loader = loaders.get("test_loader")
    # model_cls_name = model.module.__class__.__name__ if hasattr(model, "module") else model.__class__.__name__
    # log_all_epochs = model_cls_name.lower() == "conformer" # Debug for EEGConformer
    for ep in range(1, max_ep + 1):
        # should_log = (ep % 10 == 0 or ep == 1 or ep == max_ep)
        model.train(); tr_loss = []; yt = []; yp = []
        for xb, yb, _ in loaders["train_loader"]:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); logits = model(xb)
            loss = crit(logits, yb); loss.backward(); opt.step()
            tr_loss.append(loss.item())
            yt.extend(yb.cpu().numpy()); yp.extend(torch.argmax(logits, 1).cpu().numpy())
        tr_acc = accuracy_score(yt, yp); tr_loss = float(np.mean(tr_loss))

        if has_val:
            vl_loss = []; yv = []; pv = []
            model.eval()
            with torch.no_grad():
                for xb, yb, _ in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb); loss = crit(logits, yb)
                    vl_loss.append(loss.item())
                    yv.extend(yb.cpu().numpy()); pv.extend(torch.argmax(logits, 1).cpu().numpy())
            vl_acc = accuracy_score(yv, pv)
            vl_loss = float(np.mean(vl_loss))
        else:
            vl_acc = None
            vl_loss = None

        test_acc = None
        # if test_loader is not None and should_log:
        te_true, te_pred = [], []
        with torch.no_grad():
            for xb, yb, _ in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                te_true.extend(yb.cpu().numpy())
                te_pred.extend(torch.argmax(logits, 1).cpu().numpy())
        test_acc = accuracy_score(te_true, te_pred) if te_true else 0.0
        
        # if split_name and (log_all_epochs or ep <= 10):
            # log_epoch_predictions(log, split_name, ep, te_true, te_pred)

        # if should_log:
        msg = f"[Epoch {ep:03d}] train_loss={tr_loss:.4f} acc={tr_acc:.4f}"
        if has_val:
            msg += f" || val_loss={vl_loss:.4f} acc={vl_acc:.4f}"
        if test_acc is not None:
            msg += f" || test_acc={test_acc:.4f} || best_test={best_test:.4f}"
        log(msg)
        
        if test_acc > best_test :
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
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_metric

def evaluate_subjectwise(model, device, loader):
    from collections import defaultdict
    model.eval(); y_true_all = []; y_pred_all = []; subjects_all = []
    with torch.no_grad():
        for xb, yb, sb in loader:
            xb = xb.to(device)
            pred = torch.argmax(model(xb), 1).cpu().numpy()
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
