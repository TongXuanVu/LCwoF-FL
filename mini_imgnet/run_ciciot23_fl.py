import os
import sys
import time
import csv
import argparse
import random
import copy
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor

# Setup paths
import pathlib
dir_path = pathlib.Path(__file__).parent.parent.resolve()
sys.path.append(str(dir_path))

from mini_imgnet.cnn1d import CNN1DConvNet
from mini_imgnet.dataloader_ciciot23 import CICIoT23DataManager, CICIoT23Dataset
from mini_imgnet.label_remap import init_remap as _init_label_remap, remap as _remap_labels
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def convert_bn_to_gn(module, groups=8):
    """
    Thay moi BatchNorm1d bang GroupNorm. GroupNorm KHONG co running stats nen
    mien nhiem voi van de "trung binh hoa thong ke BN giua cac client non-IID".
    Chi dung khi --bn_mode gn (mac dinh giu nguyen kien truc goc).
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm1d):
            c = child.num_features
            g = groups
            while g > 1 and c % g != 0:
                g -= 1
            setattr(module, name, nn.GroupNorm(g, c))
        else:
            convert_bn_to_gn(child, groups)
    return module


class CICIoT23Model(nn.Module):
    """
    CNN1D + Dynamic Fully Connected Layer model for Class-Incremental Learning on CIC-IoT23.
    """
    # Dat = True (tu --bn_mode gn) truoc khi tao model de doi BatchNorm -> GroupNorm
    USE_GROUPNORM = False

    def __init__(self):
        super(CICIoT23Model, self).__init__()
        self.encoder = CNN1DConvNet()
        if CICIoT23Model.USE_GROUPNORM:
            convert_bn_to_gn(self.encoder)
        # Start with base task (Task 1 has 6 classes)
        self.fc = nn.Linear(self.encoder.out_dim, 6, bias=False)
        self.checkpoint_init = None
        self.l2_params = []

    def forward(self, x):
        features = self.encoder(x)
        logits = self.fc(features)
        return logits

    def expand_classifier(self, new_classes_num):
        """
        Expands the classifier weight matrix to accommodate new classes.
        """
        old_classes_num = self.fc.out_features
        weight = self.fc.weight.data.clone() # [old_classes_num, 64]
        
        # Create new fully connected layer
        new_fc = nn.Linear(self.encoder.out_dim, old_classes_num + new_classes_num, bias=False)
        
        # Copy old weights
        new_fc.weight.data[:old_classes_num] = weight
        
        # Initialize new weights orthogonally
        nn.init.orthogonal_(new_fc.weight.data[old_classes_num:])
        
        self.fc = new_fc
        print(f"[Model] Expanded classifier from {old_classes_num} to {self.fc.out_features} classes.")

    def save_base_checkpoint(self):
        """
        Saves a deep copy of base encoder weights to serve as anchor for L2 regularized parameter loss.
        """
        self.checkpoint_init = {k: v.clone().detach() for k, v in self.encoder.state_dict().items()}
        
        # Cache list of (param_object, anchor_tensor) to avoid dict lookup in forward pass
        self.l2_params = []
        for name, param in self.encoder.named_parameters():
            if name in self.checkpoint_init:
                self.l2_params.append((param, self.checkpoint_init[name]))
        print("[Model] Base checkpoint saved and L2 parameter anchors cached.")

    def L2_weight_loss(self, l2_lambda=500.0):
        """
        L2 Parameter Regularization (Anti-forgetting loss)
        Formula: L2 = l2_lambda * sum((W_base - W_current)^2)
        Optimized by utilizing the cached l2_params references.
        """
        if not self.l2_params:
            return 0.0
            
        loss = 0.0
        for param, anchor in self.l2_params:
            loss += (anchor - param).pow(2).sum()
        return loss * l2_lambda


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate(model, dataloader, device):
    model.eval()
    preds_list = []
    targets_list = []
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['data'].to(device)
            targets = batch['label'].to(device)
            
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
                
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
            preds = torch.argmax(logits, dim=1)
            
            preds_list.append(preds.cpu())
            targets_list.append(targets.cpu())
            
    if total_samples == 0:
        return {
            'acc': 0.0, 'prec_mic': 0.0, 'prec_mac': 0.0, 'prec_wei': 0.0,
            'rec_mic': 0.0, 'rec_mac': 0.0, 'rec_wei': 0.0,
            'f1_mic': 0.0, 'f1_mac': 0.0, 'f1_wei': 0.0,
            'fpr': 0.0, 'loss': 0.0
        }, np.array([]), np.array([])
        
    y_pred = torch.cat(preds_list).numpy()
    y_true = torch.cat(targets_list).numpy()
    
    acc = accuracy_score(y_true, y_pred) * 100
    
    prec_mic, rec_mic, f1_mic, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)
    prec_mac, rec_mac, f1_mac, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    prec_wei, rec_wei, f1_wei, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    # Calculate False Positive Rate (FPR)
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    fpr_per_class = []
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fpr_per_class.append(fpr)
    macro_fpr = np.mean(fpr_per_class) * 100 if len(fpr_per_class) > 0 else 0.0
    
    avg_loss = total_loss / total_samples
    
    return {
        'acc': acc,
        'prec_mic': prec_mic * 100,
        'prec_mac': prec_mac * 100,
        'prec_wei': prec_wei * 100,
        'rec_mic': rec_mic * 100,
        'rec_mac': rec_mac * 100,
        'rec_wei': rec_wei * 100,
        'f1_mic': f1_mic * 100,
        'f1_mac': f1_mac * 100,
        'f1_wei': f1_wei * 100,
        'fpr': macro_fpr,
        'loss': avg_loss
    }, y_pred, y_true


def plot_confusion_matrix(y_true, y_pred, task_id, run_dir):
    if len(y_true) == 0:
        return
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'LCwoF-FL Confusion Matrix - Task {task_id}')
    
    save_path = os.path.join(run_dir, f'confusion_matrix_task_{task_id}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved confusion matrix to: {save_path}")


def plot_confusion_matrix_round(y_true, y_pred, round_idx, task, phase, epoch, save_dir):
    if len(y_true) == 0:
        return
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'LCwoF-FL Confusion Matrix - Round {round_idx} (Task {task}, Phase {phase}, Round {epoch})')
    
    save_path = os.path.join(save_dir, f'cm_round_{round_idx}.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['state_dict']
    fc_weight = state_dict['fc.weight']
    num_classes = fc_weight.shape[0]
    
    # Expand/resize model.fc to match num_classes.
    # .to(device): nn.Linear moi mac dinh o CPU -> phai chuyen len dung device cua model,
    # neu khong se loi "mat2 is on cpu, different from ... cuda" khi eval sau resume.
    _dev = next(model.parameters()).device
    model.fc = nn.Linear(model.encoder.out_dim, num_classes, bias=False).to(_dev)
    model.load_state_dict(state_dict)
    model.to(_dev)
    
    # Restore checkpoint_init (L2 regularization anchor) and rebuild cached parameter pairs
    if 'checkpoint_init' in checkpoint and checkpoint['checkpoint_init'] is not None:
        model.checkpoint_init = checkpoint['checkpoint_init']
        model.l2_params = []
        for name, param in model.encoder.named_parameters():
            if name in model.checkpoint_init:
                model.l2_params.append((param, model.checkpoint_init[name]))
        
    return checkpoint


def _is_bn_stat_key(key):
    """running_mean / running_var / num_batches_tracked cua BatchNorm."""
    return ("running_mean" in key) or ("running_var" in key) or ("num_batches_tracked" in key)


@torch.no_grad()
def recalibrate_bn(model, loader, device, max_batches=20):
    """
    Tinh lai running stats cua BatchNorm tren server bang vai batch du lieu val.

    Ly do: FedAvg trung binh hoa running_mean/var cua cac client cuc ky non-IID
    (client 40 co 49 mau, client 22 co 622k mau) -> thong ke BN toan cuc vo nghia,
    la loi kinh dien lam FedAvg sap tren du lieu lech phan bo.
    """
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
    if not bns or loader is None:
        return
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # lay trung binh tich luy thay vi EMA
    was_training = model.training
    model.train()
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        model(batch['data'].to(device))
    if not was_training:
        model.eval()


def aggregate_fedavg(client_weights, client_sample_counts, skip_bn_stats=False):
    total_samples = sum(client_sample_counts)
    if total_samples == 0:
        return client_weights[0]

    w_avg = copy.deepcopy(client_weights[0])
    for key in w_avg.keys():
        if skip_bn_stats and _is_bn_stat_key(key):
            continue  # giu nguyen gia tri cua client dau, se duoc recalibrate sau
        w_avg[key] = torch.zeros_like(w_avg[key])

    for i in range(len(client_weights)):
        weight = client_sample_counts[i] / total_samples
        for key in w_avg.keys():
            if skip_bn_stats and _is_bn_stat_key(key):
                continue
            if torch.is_floating_point(client_weights[i][key]):
                w_avg[key] += client_weights[i][key] * weight
            else:
                if i == 0:
                    w_avg[key] = client_weights[i][key]
                else:
                    w_avg[key] = torch.max(w_avg[key], client_weights[i][key])
    return w_avg


import logging


# =============================================================================
# AFSIC-IDS aggregation, copied 1:1 from AFSIC-IDS/utils/aggregation.py
# (verbatim -- do not edit; the LCwoF wiring lives in aggregate_adaptive_robust below)
# =============================================================================
def is_aggregated_state_key(key, task, aggregate_backbone=False):
    if task == 0 or aggregate_backbone:
        return True
    if "plasticity_adapter.frozen_source" in key:
        return False
    return any(sub in key for sub in ["plasticity_adapter.adapter", "gate", "fc"])


def compute_aggregation_weights(
    args,
    global_model,
    client_accs,
    client_protos,
    client_weights,
    global_state_round_start,
    active_client_indices,
    task
):
    Q_list = []
    drift_list = []
    update_norm_list = []

    beta_acc = args.get("beta_acc", 1.0)
    beta_proto = args.get("beta_proto", 1.0)
    beta_novelty = args.get("beta_novelty", 0.5)
    beta_drift = args.get("beta_drift", 0.5)
    beta_update = args.get("beta_update", 0.2)

    for c_idx, c in enumerate(active_client_indices):
        acc_i = client_accs[c]

        # Prototype Consistency
        proto_cons_vals = []
        for class_id in range(global_model._total_classes):
            local_p = client_protos[c].get(class_id, {}).get("prototype")
            global_p = global_model.global_proto_memory.get_prototype(class_id)
            if local_p is not None and global_p is not None:
                sim = torch.sum(F.normalize(local_p, p=2, dim=0) * F.normalize(global_p, p=2, dim=0)).item()
                proto_cons_vals.append(sim)
        proto_cons_i = sum(proto_cons_vals) / len(proto_cons_vals) if proto_cons_vals else 1.0

        # Novelty
        novelty_vals = []
        new_classes = range(global_model._known_classes, global_model._total_classes)
        old_classes = range(global_model._known_classes)
        if new_classes and old_classes:
            for n_c in new_classes:
                local_p = client_protos[c].get(n_c, {}).get("prototype")
                if local_p is not None:
                    local_p = F.normalize(local_p, p=2, dim=0)
                    min_dist = 1.0
                    for o_c in old_classes:
                        global_p = global_model.global_proto_memory.get_prototype(o_c)
                        if global_p is not None:
                            global_p = F.normalize(global_p, p=2, dim=0)
                            dist = 1.0 - torch.sum(local_p * global_p).item()
                            if dist < min_dist:
                                min_dist = dist
                    novelty_vals.append(min_dist)
        novelty_i = sum(novelty_vals) / len(novelty_vals) if novelty_vals else 0.5

        # Drift and Update Norm
        drift_val = 0.0
        update_val = 0.0
        num_params = 0
        local_dict = client_weights[c_idx]
        for k in local_dict.keys():
            if is_aggregated_state_key(k, task, args.get("aggregate_backbone", False)):
                diff = local_dict[k].float() - global_state_round_start[k].float()
                drift_val += torch.sum(diff ** 2).item()
                update_val += torch.sum(diff ** 2).item()
                num_params += diff.numel()

        drift_i = np.sqrt(drift_val / max(1, num_params))
        update_norm_i = np.sqrt(update_val / max(1, num_params))

        Q_i = beta_acc * acc_i + beta_proto * proto_cons_i + beta_novelty * novelty_i - beta_drift * drift_i - beta_update * update_norm_i
        Q_list.append(Q_i)
        drift_list.append(drift_i)
        update_norm_list.append(update_norm_i)

        logging.info(
            f"Client {c} => Q_i: {Q_i:.4f} | Acc: {acc_i*100:.2f}% | "
            f"ProtoCons: {proto_cons_i:.4f} | Novelty: {novelty_i:.4f} | "
            f"Drift: {drift_i:.4f} | UpdateNorm: {update_norm_i:.4f}"
        )

    accepted_positions = list(range(len(Q_list)))
    if args.get("robust_filter_updates", True) and len(Q_list) > 2:
        update_arr = np.array(update_norm_list, dtype=np.float64)
        drift_arr = np.array(drift_list, dtype=np.float64)
        update_med = float(np.median(update_arr))
        drift_med = float(np.median(drift_arr))
        update_mad = float(np.median(np.abs(update_arr - update_med))) + 1e-8
        drift_mad = float(np.median(np.abs(drift_arr - drift_med))) + 1e-8
        z_limit = args.get("robust_z", 3.5)
        max_update_norm = args.get("max_update_norm", None)
        accepted_positions = []
        for pos, c in enumerate(active_client_indices):
            update_ok = (update_arr[pos] - update_med) / update_mad <= z_limit
            drift_ok = (drift_arr[pos] - drift_med) / drift_mad <= z_limit
            norm_ok = max_update_norm is None or update_arr[pos] <= float(max_update_norm)
            if update_ok and drift_ok and norm_ok:
                accepted_positions.append(pos)
            else:
                logging.warning(
                    f"Client {c} rejected by robust filter | "
                    f"Drift: {drift_arr[pos]:.4f} | UpdateNorm: {update_arr[pos]:.4f}"
                )
        if not accepted_positions:
            logging.warning("Robust filter rejected all clients; falling back to all active clients.")
            accepted_positions = list(range(len(Q_list)))

    Q_accepted = [Q_list[pos] for pos in accepted_positions]
    Q_tensor = torch.tensor(Q_accepted, dtype=torch.float32)
    tau_agg = args.get("tau_aggregation", 1.0)
    alpha = torch.softmax(Q_tensor / tau_agg, dim=0).tolist()

    return alpha, accepted_positions, Q_accepted


# =============================================================================
# LCwoF wiring for the verbatim AFSIC aggregation above.
# LCwoF has no prototype memory / adapters, so `client_protos` are empty: AFSIC's own
# fallbacks then set proto_cons=1.0 and novelty=0.5 for EVERY client (constants that cancel
# inside the softmax). The effective score is Q_i = beta_acc*acc_i - beta_drift*drift_i
# - beta_update*update_norm_i, with acc_i = server-side validation accuracy of client i.
# =============================================================================
class _ProtoMemStub:
    def get_prototype(self, class_id):
        return None


class _GlobalShim:
    """Minimal stand-in so the verbatim AFSIC function runs on LCwoF (no prototype memory)."""
    def __init__(self, total_classes, known_classes):
        self._total_classes = total_classes
        self._known_classes = known_classes
        self.global_proto_memory = _ProtoMemStub()


# AFSIC-IDS default aggregation hyper-parameters (utils/aggregation.py defaults).
# aggregate_backbone=True: LCwoF trains the whole network (no frozen backbone / adapters),
# so aggregate every parameter -- the same set FedAvg averages -- for a fair robust-vs-FedAvg test.
_AFSIC_AGG_ARGS = {
    "beta_acc": 1.0,
    "beta_proto": 1.0,
    "beta_novelty": 0.5,
    "beta_drift": 0.5,
    "beta_update": 0.2,
    "tau_aggregation": 1.0,
    "robust_z": 3.5,
    "robust_filter_updates": True,
    "max_update_norm": None,
    "aggregate_backbone": True,
}


@torch.no_grad()
def _client_val_acc(eval_model, client_state, val_batches, num_classes):
    """Overall validation accuracy of a client model in [0,1] (mirrors AFSIC _compute_accuracy 'total')."""
    eval_model.load_state_dict(client_state)
    eval_model.eval()
    correct = 0
    total = 0
    for inputs, y_true in val_batches:
        logits = eval_model(inputs)[:, :num_classes]
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        correct += int((preds == y_true).sum())
        total += len(y_true)
    return (correct / total) if total > 0 else 0.0


def aggregate_adaptive_robust(client_weights, client_sample_counts, global_weights,
                              eval_model=None, val_loader=None, device=None,
                              num_eval_classes=None, task=1, max_eval_batches=4):
    """
    AFSIC-IDS aggregation ported 1:1 onto LCwoF-FL: verbatim `compute_aggregation_weights`
    (Q-score + MAD robust filter + softmax) and the verbatim alpha-weighted averaging from
    AFSIC's trainer. `client_sample_counts` is unused (AFSIC weights by quality, not volume);
    kept for a drop-in signature with aggregate_fedavg.
    """
    num_clients = len(client_weights)
    # Round-start global snapshot (state_dict returns live refs that per-client eval overwrites).
    global_snapshot = {k: v.detach().clone() for k, v in global_weights.items()}

    # Server-side validation accuracy per client (AFSIC Acc_{i,val}).
    if (eval_model is not None) and (val_loader is not None) and (num_eval_classes is not None):
        val_batches = []
        for bi, batch in enumerate(val_loader):
            if bi >= max_eval_batches:
                break
            val_batches.append((batch['data'].to(device), batch['label'].numpy()))
        client_accs = [_client_val_acc(eval_model, client_weights[i], val_batches, num_eval_classes)
                       for i in range(num_clients)]
        eval_model.load_state_dict(global_snapshot)
    else:
        client_accs = [0.0] * num_clients

    active_client_indices = list(range(num_clients))
    client_protos = [{} for _ in range(num_clients)]
    shim = _GlobalShim(total_classes=(num_eval_classes or 0), known_classes=0)

    alpha, accepted_positions, _ = compute_aggregation_weights(
        args=_AFSIC_AGG_ARGS,
        global_model=shim,
        client_accs=client_accs,
        client_protos=client_protos,
        client_weights=client_weights,
        global_state_round_start=global_snapshot,
        active_client_indices=active_client_indices,
        task=task,
    )

    client_weights_accepted = [client_weights[pos] for pos in accepted_positions]

    print(f"      [AFSIC-IDS Aggregation] Acc: {[round(a, 3) for a in client_accs]} | "
          f"Alphas: {[round(a, 4) for a in alpha]}")

    # Verbatim AFSIC quality-aware weighted averaging (trainer.py).
    global_dict = copy.deepcopy(global_snapshot)
    aggregate_backbone = _AFSIC_AGG_ARGS.get("aggregate_backbone", False)
    for k in global_dict.keys():
        if is_aggregated_state_key(k, task, aggregate_backbone):
            val = client_weights_accepted[0][k].float() * alpha[0]
            for c_idx in range(1, len(client_weights_accepted)):
                val += client_weights_accepted[c_idx][k].float() * alpha[c_idx]
            global_dict[k] = val.to(global_dict[k].dtype)
    return global_dict


def client_select_exemplars(exemplar_memory, x, y, classes, m_per_class=20):
    """
    Select exemplars locally on the client and store them in exemplar_memory.
    """
    for c in classes:
        class_mask = (y == c)
        x_c = x[class_mask]
        y_c = y[class_mask]
        
        n_samples = x_c.shape[0]
        if n_samples == 0:
            continue
            
        n_select = max(1, int(n_samples * 0.01))
        indices = np.random.choice(n_samples, n_select, replace=False)
        exemplar_memory[c] = {
            'x': x_c[indices],
            'y': y_c[indices]
        }


def client_get_calibration_dataset(seen_classes, current_task_x, current_task_y, current_task_classes, exemplar_memory):
    """
    Construct calibration dataset locally on the client.
    """
    cal_x_list = []
    cal_y_list = []

    # 1. Old classes from local exemplar memory
    for c in seen_classes:
        if c not in current_task_classes:
            if c in exemplar_memory:
                cal_x_list.append(exemplar_memory[c]['x'].cpu())
                cal_y_list.append(exemplar_memory[c]['y'].cpu())

    # 2. Current classes from local training data
    for c in current_task_classes:
        class_mask = (current_task_y == c)
        x_c = current_task_x[class_mask]
        y_c = current_task_y[class_mask]
        
        n_samples = x_c.shape[0]
        if n_samples > 0:
            n_select = max(1, int(n_samples * 0.01))
            indices = np.random.choice(n_samples, n_select, replace=False)
            cal_x_list.append(x_c[indices].cpu())
            cal_y_list.append(y_c[indices].cpu())

    if len(cal_x_list) == 0:
        return None

    cal_x = torch.cat(cal_x_list, dim=0)
    cal_y = torch.cat(cal_y_list, dim=0)
    return CICIoT23Dataset(cal_x, cal_y)


def main():
    parser = argparse.ArgumentParser(description="LCwoF training in Federated Learning scenario")
    parser.add_argument("--data_root", type=str, default="C:/FederatedLearning/FL/core/data_split",
                        help="Path to the directory containing global_test_data.pt and federated_data")
    parser.add_argument("--total_clients", type=int, default=100,
                        help="Tong so client trong partition (100 cho bo data '100 client', 10 cho bo cu). "
                             "So client active moi task = 50%%..100%% cua gia tri nay.")
    parser.add_argument("--bn_mode", type=str, default="avg", choices=["avg", "recalib", "gn"],
                        help="Xu ly BatchNorm khi aggregate: avg = trung binh ca running stats (goc); "
                             "recalib = khong trung binh stats, tinh lai tren server; "
                             "gn = thay BatchNorm1d bang GroupNorm (bat bien voi non-IID).")
    parser.add_argument("--l2_lambda", type=float, default=500.0,
                        help="Lambda coefficient for L2 parameter regularization (default 500.0)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default 0.001)")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size (default 1024)")
    parser.add_argument("--epochs_base", type=int, default=30, help="Communication rounds for base task (default 30)")
    parser.add_argument("--epochs_novel", type=int, default=30, help="Communication rounds for novel tasks Phase 2 (default 30)")
    parser.add_argument("--epochs_crt", type=int, default=30, help="Communication rounds for calibration training Phase 3 (default 30)")
    parser.add_argument("--local_epochs", type=int, default=1, help="Local training epochs on client per round (default 1)")
    parser.add_argument("--memory_per_class", type=int, default=20, help="Exemplar memory budget per class (default 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    parser.add_argument("--debug", action="store_true", help="Debug mode: fast training with 2 rounds")
    parser.add_argument("--max_test_samples_per_class", type=int, default=0,
                        help="So mau test toi da moi lop. 0 (mac dinh) = KHONG gioi han, danh gia tren toan bo tap test cua cac lop da hoc.")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test", "resume"],
                        help="Execution mode: train (default), test (evaluate checkpoints), or resume (resume training)")
    parser.add_argument("--resume_path", type=str, default="",
                        help="Path to a model checkpoint (.pt) to resume from or to test")
    parser.add_argument("--test_dir", type=str, default="",
                        help="Path to directory containing multiple checkpoints to evaluate in test mode")
    parser.add_argument("--run_dir", type=str, default="",
                        help="Custom directory to save logs, CSV, and checkpoints")
    parser.add_argument("--use_fewshot", action="store_true",
                        help="Use the few-shot data split for incremental tasks")
    parser.add_argument("--use_10shot", action="store_true",
                        help="Use the 10-shot data split for incremental tasks")
    parser.add_argument("--novel_data_dir", type=str, default="",
                        help="Custom directory name for novel tasks federated data. Overrides other flags if provided.")
    parser.add_argument("--aggregation", type=str, default="robust", choices=["fedavg", "robust"],
                        help="Aggregation method to use at the server (default: robust)")

    args = parser.parse_args()
    
    if args.debug:
        print("[DEBUG] Debug mode activated! Adjusting rounds to 2 and test sampling to 200.")
        args.epochs_base = 2
        args.epochs_novel = 2
        args.epochs_crt = 2
        args.max_test_samples_per_class = 200
        
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using device: {device}")

    # <=0 nghia la KHONG gioi han: danh gia tren TOAN BO mau test cua cac lop da hoc
    _test_cap = None if args.max_test_samples_per_class <= 0 else args.max_test_samples_per_class
    print(f"[Eval] Test set: {'TOAN BO mau cua cac lop da hoc' if _test_cap is None else str(_test_cap) + ' mau/lop'}")

    # Che do xu ly BatchNorm khi aggregate (chan doan FedAvg tren du lieu non-IID)
    if args.bn_mode == "gn":
        CICIoT23Model.USE_GROUPNORM = True
        print("[BN] bn_mode=gn -> thay BatchNorm1d bang GroupNorm (khong co running stats).")
    elif args.bn_mode == "recalib":
        print("[BN] bn_mode=recalib -> KHONG trung binh running stats; tinh lai tren server moi round.")
    else:
        print("[BN] bn_mode=avg -> trung binh ca running stats (hanh vi goc).")
    
    # 1. Setup Global Test Data Manager
    dm = CICIoT23DataManager(data_root=args.data_root)
    
    # Define tasks and their corresponding classes
    tasks_classes = [
        [0, 1, 2, 3, 4, 5],       # Task 1 (Base task)
        [6, 7, 8, 9, 10, 11],     # Task 2
        [12, 13, 14, 15, 16, 17], # Task 3
        [18, 19, 20, 21, 22, 23], # Task 4
        [24, 25, 26, 27, 28],     # Task 5
        [29, 30, 31, 32, 33]      # Task 6
    ]
    
    # Initialize Global Model
    global_model = CICIoT23Model().to(device)
    
    # Client memories list: 10 clients (0 to 9)
    # Mot ban ghi nho exemplar cho MOI client trong partition (100 client, khong phai 10)
    client_memories = [{} for _ in range(args.total_clients)]
    
    # -------------------------------------------------------------------------
    # MODE: TEST
    # -------------------------------------------------------------------------
    if args.mode == "test":
        print("\n" + "="*80)
        print("RUNNING IN TEST MODE (EVALUATING CHECKPOINTS)")
        print("="*80)
        
        checkpoints_to_test = []
        if args.resume_path:
            checkpoints_to_test.append(args.resume_path)
        elif args.test_dir:
            import glob
            pt_files = glob.glob(os.path.join(args.test_dir, "*.pt")) + glob.glob(os.path.join(args.test_dir, "checkpoints", "*.pt"))
            
            def cp_key(path):
                name = os.path.basename(path)
                if "checkpoint_task" in name:
                    try:
                        return (0, int(name.split("_")[-1].split(".")[0]), 0, 0)
                    except:
                        return (0, 999, 0, 0)
                elif "epoch" in name or "round" in name:
                    try:
                        parts = name.split("_")
                        t = int(parts[1])
                        if "phase" in name:
                            p = int(parts[2].replace("phase", ""))
                            e = int(parts[4].split(".")[0])
                        else:
                            p = 1
                            e = int(parts[3].split(".")[0])
                        return (1, t, p, e)
                    except:
                        return (1, 999, 999, 999)
                return (2, name)
                
            checkpoints_to_test = sorted(pt_files, key=cp_key)
        else:
            print("[Error] In test mode, you must provide either --resume_path or --test_dir!")
            return
            
        if not checkpoints_to_test:
            print(f"[Warning] No checkpoint files (.pt) found!")
            return
            
        print(f"Found {len(checkpoints_to_test)} checkpoints to evaluate:")
        print(f"%-40s | %-12s | %-12s | %-12s" % ("Checkpoint File", "Accuracy", "F1-Macro", "Seen Classes"))
        print("-" * 85)
        
        for cp_path in checkpoints_to_test:
            if not os.path.exists(cp_path):
                print(f"[Warning] File not found: {cp_path}")
                continue
                
            try:
                checkpoint = load_checkpoint(global_model, cp_path, device)
                if 'seen_classes' in checkpoint:
                    seen_classes = checkpoint['seen_classes']
                else:
                    num_classes = global_model.fc.out_features
                    seen_classes = []
                    for tc in tasks_classes:
                        if len(seen_classes) >= num_classes:
                            break
                        seen_classes.extend(tc)
                    seen_classes = seen_classes[:num_classes]
                    
                test_dataset = dm.get_test_dataset(seen_classes, max_samples_per_class=_test_cap)
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
                metrics, _, _ = evaluate(global_model, test_loader, device)
                
                print("%-40s | %10.2f%% | %10.2f%% | %d classes" % (
                    os.path.basename(cp_path),
                    metrics['acc'],
                    metrics['f1_mac'],
                    len(seen_classes)
                ))
            except Exception as e:
                print(f"[Error] Failed to evaluate {os.path.basename(cp_path)}: {str(e)}")
                
        print("="*80 + "\n")
        return

    # -------------------------------------------------------------------------
    # INITIALIZATION FOR TRAINING / RESUME
    # -------------------------------------------------------------------------
    start_task = 1
    start_epoch = 0
    start_phase = 1 # 1: Base pretraining, 2: Novel explicit, 3: Calibration CRT
    seen_classes = []
    
    if args.mode == "resume":
        if not args.resume_path or not os.path.exists(args.resume_path):
            print(f"[Error] In resume mode, you must provide a valid --resume_path!")
            return
            
        print("\n" + "="*80)
        print(f"RESUMING TRAINING FROM CHECKPOINT: {args.resume_path}")
        print("="*80)
        
        checkpoint = load_checkpoint(global_model, args.resume_path, device)
        
        if 'seen_classes' in checkpoint:
            seen_classes = checkpoint['seen_classes']
        else:
            num_classes = global_model.fc.out_features
            seen_classes = []
            for tc in tasks_classes:
                if len(seen_classes) >= num_classes:
                    break
                seen_classes.extend(tc)
            seen_classes = seen_classes[:num_classes]
            
        if 'client_memories' in checkpoint:
            client_memories = checkpoint['client_memories']
            # Checkpoint cu co the it client hon partition hien tai -> bu them cho du
            while len(client_memories) < args.total_clients:
                client_memories.append({})
            print(f"[DataManager] Restored exemplar memory containing {len(client_memories)} clients.")
            
        filename = os.path.basename(args.resume_path)
        if "checkpoint_task" in filename:
            task_idx_completed = checkpoint.get('task_idx', 1)
            start_task = task_idx_completed + 1
            start_epoch = 0
            start_phase = 2 if start_task > 1 else 1
            print(f"[Resume] Final Task {task_idx_completed} checkpoint detected. Resuming from Task {start_task} (Pha {start_phase}, Round 1).")
        elif "round" in filename or "epoch" in filename or "latest" in filename:
            # 'latest' = file resume rolling (resume_latest.pt) ghi de moi round.
            # Doc task/phase/epoch tu payload trong checkpoint.
            start_task = checkpoint.get('task_idx', 1)
            start_epoch = checkpoint.get('epoch', 0)
            start_phase = checkpoint.get('phase', 1)
            print(f"[Resume] Round checkpoint detected. Resuming Task {start_task} (Pha {start_phase}) starting at Round {start_epoch + 1}.")
        else:
            # Fallback an toan: neu ten file khong ro, van doc tu payload thay vi crash.
            start_task = checkpoint.get('task_idx', 1)
            start_epoch = checkpoint.get('epoch', 0)
            start_phase = checkpoint.get('phase', 1)
            print(f"[Resume] Unknown checkpoint name, doc tu payload: Task {start_task} Pha {start_phase} Round {start_epoch + 1}.")
            
        run_dir = os.path.dirname(os.path.abspath(args.resume_path))
        if os.path.basename(run_dir) == "checkpoints":
            run_dir = os.path.dirname(run_dir)
            
        print(f"[Logs] Output directory (reused): {run_dir}")
        csv_path = os.path.join(run_dir, "metrics.csv")
        csv_file = open(csv_path, "a", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        
        round_csv_path = os.path.join(run_dir, "round_metrics.csv")
        round_csv_file = open(round_csv_path, "a", newline="", encoding="utf-8")
        round_csv_writer = csv.writer(round_csv_file)
        
    else: # train mode
        if args.run_dir:
            run_dir = args.run_dir
        else:
            timestamp = datetime.now().strftime("%d-%m-%y_%H-%M")
            run_dir = os.path.join(str(dir_path), "logs", "lcwof_fl_ciciot23", f"seed{args.seed}_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        print(f"[Logs] Output directory: {run_dir}")
        
        csv_path = os.path.join(run_dir, "metrics.csv")
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "task", "acc", "prec_mic", "prec_mac", "prec_wei",
            "rec_mic", "rec_mac", "rec_wei", "f1_mic", "f1_mac", "f1_wei"
        ])
        
        round_csv_path = os.path.join(run_dir, "round_metrics.csv")
        round_csv_file = open(round_csv_path, "w", newline="", encoding="utf-8")
        round_csv_writer = csv.writer(round_csv_file)
        round_csv_writer.writerow([
            "round_idx", "task", "phase", "round", "train_loss", "test_loss",
            "acc", "prec_mic", "prec_mac", "prec_wei",
            "rec_mic", "rec_mac", "rec_wei",
            "f1_mic", "f1_mac", "f1_wei", "fpr"
        ])

    # =========================================================================
    # TASK 1: Base Task training
    # =========================================================================
    if start_task == 1:
        task_idx = 1
        classes = tasks_classes[0]
        if not seen_classes:
            seen_classes.extend(classes)
            
        # Lich client tang dan theo TY LE 50% -> 100% (10 client: 5..10 | 100 client: 50..100)
        num_clients = int(round(args.total_clients * 0.5))
        print("\n" + "="*80)
        print(f"STARTING TASK 1 - FEDERATED (Base classes {classes}, Clients: {num_clients})")
        print("="*80)
        
        # Load local client datasets
        client_datasets = []
        for c in range(num_clients):
            # Ho tro ca 3 layout: <root>/federated_data/, <root>/data/federated_data/,
            # va layout PHANG <root>/*.pt (Kaggle dataset khong giu thu muc con)
            for _cand in (
                os.path.join(args.data_root, "federated_data", f"client_{c}_task_1.pt"),
                os.path.join(args.data_root, "data", "federated_data", f"client_{c}_task_1.pt"),
                os.path.join(args.data_root, f"client_{c}_task_1.pt"),
            ):
                c_path = _cand
                if os.path.exists(c_path):
                    break
            if not os.path.exists(c_path):
                # Voi partition 100-client, mot so client KHONG co du lieu o task nay
                # (vd task 1 chi 47/50 client co file). Bo qua an toan.
                client_datasets.append((None, None))
                print(f"  Client {c}: khong co file task 1 -> bo qua.")
                continue
            c_data = torch.load(c_path, map_location="cpu", weights_only=False)
            c_x, c_y = c_data["x"].float(), _remap_labels(c_data["y"].long())
            if args.debug:
                c_x = c_x[:500]
                c_y = c_y[:500]
            client_datasets.append((c_x, c_y))
            print(f"  Client {c} loaded: {len(c_x)} training samples.")

        # Chi train tren cac client THUC SU co du lieu
        active_clients = [c for c in range(num_clients)
                          if client_datasets[c][0] is not None and len(client_datasets[c][0]) > 0]
        if not active_clients:
            raise RuntimeError("Task 1: khong co client nao co du lieu. Kiem tra lai --data_root.")
        print(f"  -> Task 1: {len(active_clients)}/{num_clients} client co du lieu.")

        # Pre-create DataLoaders for all clients to avoid creation overhead in round loop
        client_loaders = {}
        for c in active_clients:
            c_x, c_y = client_datasets[c]
            train_dataset = CICIoT23Dataset(c_x, c_y)
            client_loaders[c] = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        # Small held-out val set reused every round for server-side quality scoring
        val_dataset_agg = dm.get_test_dataset(seen_classes, max_samples_per_class=200)
        val_loader_agg = DataLoader(val_dataset_agg, batch_size=args.batch_size, shuffle=False)

        initial_epoch = start_epoch if (start_task == 1) else 0

        for epoch in range(initial_epoch, args.epochs_base):
            # Define client local training worker for ThreadPoolExecutor
            def train_local_client_task1(c):
                # Optimize PyTorch CPU execution inside worker
                torch.set_num_threads(1)
                
                train_loader = client_loaders[c]
                c_x, _ = client_datasets[c]
                
                # Instantiate model and load state_dict (much faster than deepcopy)
                local_model = CICIoT23Model().to(device)
                local_model.fc = nn.Linear(local_model.encoder.out_dim, global_model.fc.out_features, bias=False).to(device)
                local_model.load_state_dict(global_model.state_dict())
                local_model.train()
                
                optimizer = torch.optim.Adam(local_model.parameters(), lr=args.lr, weight_decay=1e-4)
                
                losses = 0.0
                for local_ep in range(args.local_epochs):
                    for batch in train_loader:
                        inputs = batch['data'].to(device)
                        targets = batch['label'].to(device)
                        
                        optimizer.zero_grad()
                        logits = local_model(inputs)
                        loss = F.cross_entropy(logits, targets)
                            
                        loss.backward()
                        optimizer.step()
                        
                        losses += loss.item()
                        
                avg_l = losses / (len(train_loader) * args.local_epochs)
                return local_model.state_dict(), len(c_x), avg_l

            # Execute client training concurrently (chi cac client co du lieu)
            with ThreadPoolExecutor(max_workers=min(len(active_clients), 16)) as executor:
                results = list(executor.map(train_local_client_task1, active_clients))
                
            client_weights = [r[0] for r in results]
            client_sample_counts = [r[1] for r in results]
            client_losses = [r[2] for r in results]
                
            # Server FedAvg aggregation
            avg_loss = np.mean(client_losses)
            if args.aggregation == "fedavg":
                aggregated_weights = aggregate_fedavg(client_weights, client_sample_counts,
                                                      skip_bn_stats=(args.bn_mode == 'recalib'))
            else:
                aggregated_weights = aggregate_adaptive_robust(
                    client_weights, client_sample_counts, global_model.state_dict(),
                    eval_model=global_model, val_loader=val_loader_agg, device=device,
                    num_eval_classes=len(seen_classes))
            global_model.load_state_dict(aggregated_weights)
            if args.bn_mode == 'recalib':
                recalibrate_bn(global_model, val_loader_agg, device)

            print(f"Task 1 Round {epoch+1}/{args.epochs_base} => Avg Client Loss: {avg_loss:.4f}")
            
            # Server Evaluation
            test_dataset = dm.get_test_dataset(seen_classes, max_samples_per_class=_test_cap)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            metrics_round, y_pred, y_true = evaluate(global_model, test_loader, device)
            
            round_idx = epoch + 1
            round_csv_writer.writerow([
                round_idx, 1, 1, epoch + 1, round(avg_loss, 4), round(metrics_round['loss'], 4),
                round(metrics_round['acc'], 4),
                round(metrics_round['prec_mic'], 4), round(metrics_round['prec_mac'], 4), round(metrics_round['prec_wei'], 4),
                round(metrics_round['rec_mic'], 4),  round(metrics_round['rec_mac'], 4),  round(metrics_round['rec_wei'], 4),
                round(metrics_round['f1_mic'], 4),   round(metrics_round['f1_mac'], 4),   round(metrics_round['f1_wei'], 4),
                round(metrics_round['fpr'], 4)
            ])
            round_csv_file.flush()
            
            # Save confusion matrix for this round
            cm_dir = os.path.join(run_dir, "confusion_matrices")
            os.makedirs(cm_dir, exist_ok=True)
            plot_confusion_matrix_round(y_true, y_pred, round_idx, 1, 1, epoch + 1, cm_dir)
            
            # Save epoch checkpoint
            epoch_checkpoint_path = os.path.join(run_dir, "checkpoints", "resume_latest.pt")
            os.makedirs(os.path.dirname(epoch_checkpoint_path), exist_ok=True)
            torch.save({
                'task_idx': 1,
                'epoch': epoch + 1,
                'state_dict': global_model.state_dict(),
                'checkpoint_init': global_model.checkpoint_init,
                'seen_classes': seen_classes,
                'client_memories': client_memories,
            }, epoch_checkpoint_path)
            
        # Clients select exemplars locally (bo qua client khong co du lieu)
        for c in active_clients:
            c_x, c_y = client_datasets[c]
            client_select_exemplars(client_memories[c], c_x, c_y, classes, m_per_class=args.memory_per_class)
            
        global_model.save_base_checkpoint()
        
        test_dataset = dm.get_test_dataset(seen_classes, max_samples_per_class=_test_cap)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        metrics, y_pred, y_true = evaluate(global_model, test_loader, device)
        
        print(f"\nTask 1 Final Metrics => Acc: {metrics['acc']:.2f}%, F1-Macro: {metrics['f1_mac']:.2f}%")
        csv_writer.writerow([
            task_idx, round(metrics['acc'], 4),
            round(metrics['prec_mic'], 4), round(metrics['prec_mac'], 4), round(metrics['prec_wei'], 4),
            round(metrics['rec_mic'], 4),  round(metrics['rec_mac'], 4),  round(metrics['rec_wei'], 4),
            round(metrics['f1_mic'], 4),   round(metrics['f1_mac'], 4),   round(metrics['f1_wei'], 4)
        ])
        csv_file.flush()
        plot_confusion_matrix(y_true, y_pred, task_idx, run_dir)
        
        task_checkpoint_path = os.path.join(run_dir, f"checkpoint_task_{task_idx}.pt")
        torch.save({
            'task_idx': task_idx,
            'state_dict': global_model.state_dict(),
            'seen_classes': seen_classes,
            'client_memories': client_memories,
            'checkpoint_init': global_model.checkpoint_init,
        }, task_checkpoint_path)
        print(f"[Checkpoint] Saved final task checkpoint to: {task_checkpoint_path}")
        
        start_phase = 2

    # =========================================================================
    # TASKS 2 to 6: Incremental Tasks
    # =========================================================================
    next_task = max(2, start_task)
    for task_idx in range(next_task, 7):
        classes = tasks_classes[task_idx - 1]
        for c in classes:
            if c not in seen_classes:
                seen_classes.append(c)
                
        # Task 2..6 -> 60%, 70%, 80%, 90%, 100% tong so client
        num_clients = int(round(args.total_clients * (0.4 + 0.1 * task_idx)))
        print("\n" + "="*80)
        print(f"STARTING TASK {task_idx} - FEDERATED (Novel classes {classes}, Clients: {num_clients})")
        print("="*80)

        # Small held-out val set (all seen classes) reused for server-side quality scoring
        val_dataset_agg = dm.get_test_dataset(seen_classes, max_samples_per_class=200)
        val_loader_agg = DataLoader(val_dataset_agg, batch_size=args.batch_size, shuffle=False)
        
        # Load local client datasets
        client_datasets = []
        
        novel_data_dir = "federated_data"
        if args.novel_data_dir:
            novel_data_dir = args.novel_data_dir
        elif args.use_fewshot:
            novel_data_dir = "federated_data_fewshot"
        elif args.use_10shot:
            novel_data_dir = "federated_data_10shot"
            
        for c in range(num_clients):
            c_path = os.path.join(args.data_root, novel_data_dir, f"client_{c}_task_{task_idx}.pt")
            if not os.path.exists(c_path):
                # layout PHANG (Kaggle dataset)
                c_path = os.path.join(args.data_root, f"client_{c}_task_{task_idx}.pt")
            if os.path.exists(c_path):
                c_data = torch.load(c_path, map_location="cpu", weights_only=False)
                c_x, c_y = c_data["x"].float(), _remap_labels(c_data["y"].long())
                if args.debug:
                    c_x = c_x[:500]
                    c_y = c_y[:500]
                client_datasets.append((c_x, c_y))
            else:
                client_datasets.append((None, None))
                
        expected_classes = len(seen_classes)
        if global_model.fc.out_features < expected_classes:
            global_model.expand_classifier(len(classes))
            global_model.to(device)
            
            print("[Server] Initializing new classifier weights using aggregated mean feature prototypes...")
            # Aggregate prototype sums and counts across clients
            sum_features_global = {c_val: torch.zeros(global_model.encoder.out_dim, device=device) for c_val in classes}
            count_features_global = {c_val: 0 for c_val in classes}
            
            for c in range(num_clients):
                c_x, c_y = client_datasets[c]
                if c_x is None:
                    continue
                    
                local_model = CICIoT23Model().to(device)
                local_model.fc = nn.Linear(local_model.encoder.out_dim, global_model.fc.out_features, bias=False).to(device)
                local_model.load_state_dict(global_model.state_dict())
                local_model.eval()
                
                with torch.no_grad():
                    for class_val in classes:
                        c_mask = (c_y == class_val)
                        x_c = c_x[c_mask]
                        total_c = len(x_c)
                        if total_c > 0:
                            sum_features = torch.zeros(local_model.encoder.out_dim, device=device)
                            eval_batch_size = 8192
                            for i in range(0, total_c, eval_batch_size):
                                batch_x = x_c[i:i + eval_batch_size].to(device)
                                features = local_model.encoder(batch_x)
                                sum_features += features.sum(dim=0)
                            
                            sum_features_global[class_val] += sum_features
                            count_features_global[class_val] += total_c
                            
            # Server updates classifier weights
            with torch.no_grad():
                for class_val in classes:
                    g_count = count_features_global[class_val]
                    if g_count > 0:
                        mean_feature = sum_features_global[class_val] / g_count
                        mean_feature = F.normalize(mean_feature, p=2, dim=0)
                        global_model.fc.weight.data[class_val] = mean_feature
                        print(f"  Initialized class {class_val} weights with {g_count} samples.")
                        
        # ---------------------------------------------------------------------
        # PHASE 2: Explicit learning of novel classes (with L2 parameter loss)
        # ---------------------------------------------------------------------
        run_phase2 = (task_idx > start_task) or (start_phase <= 2)
        
        if run_phase2:
            print(f"\n--- [Phase 2 - Federated] Training novel classes for Task {task_idx} ---")
            
            # Pre-create DataLoaders for all clients to avoid creation overhead in round loop
            client_loaders = []
            for c in range(num_clients):
                c_x, c_y = client_datasets[c]
                if c_x is not None:
                    train_dataset = CICIoT23Dataset(c_x, c_y)
                    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
                    client_loaders.append(train_loader)
                else:
                    client_loaders.append(None)
            
            initial_epoch = start_epoch if (task_idx == start_task and start_phase == 2) else 0
            
            for epoch in range(initial_epoch, args.epochs_novel):
                # Worker definition for concurrent execution of Phase 2
                def train_local_client_phase2(c):
                    torch.set_num_threads(1)
                    
                    train_loader = client_loaders[c]
                    c_x, _ = client_datasets[c]
                    
                    local_model = CICIoT23Model().to(device)
                    local_model.fc = nn.Linear(local_model.encoder.out_dim, global_model.fc.out_features, bias=False).to(device)
                    local_model.load_state_dict(global_model.state_dict())
                    local_model.checkpoint_init = global_model.checkpoint_init
                    
                    # Bind pre-cached L2 param references to local parameters
                    if global_model.l2_params:
                        local_model.l2_params = []
                        for name, param in local_model.encoder.named_parameters():
                            if name in local_model.checkpoint_init:
                                local_model.l2_params.append((param, local_model.checkpoint_init[name]))
                                
                    local_model.train()
                    optimizer = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, local_model.parameters()), 
                        lr=args.lr, 
                        weight_decay=1e-4
                    )
                    losses = 0.0
                    for local_ep in range(args.local_epochs):
                        for batch in train_loader:
                            inputs = batch['data'].to(device)
                            targets = batch['label'].to(device)
                            
                            optimizer.zero_grad()
                            logits = local_model(inputs)
                            local_logits = logits[:, classes]
                            local_targets = torch.tensor([classes.index(t.item()) for t in targets], dtype=torch.long, device=device)
                            
                            cls_loss = F.cross_entropy(local_logits, local_targets)
                                
                            l2_loss = local_model.L2_weight_loss(l2_lambda=args.l2_lambda)
                            total_loss = cls_loss + l2_loss
                                
                            total_loss.backward()
                            optimizer.step()
                            
                            losses += total_loss.item()
                            
                    avg_l = losses / (len(train_loader) * args.local_epochs)
                    return local_model.state_dict(), len(c_x), avg_l

                # Run parallel training
                active_clients = [c for c in range(num_clients) if client_datasets[c][0] is not None]
                with ThreadPoolExecutor(max_workers=len(active_clients)) as executor:
                    results = list(executor.map(train_local_client_phase2, active_clients))
                    
                client_weights = [r[0] for r in results]
                client_sample_counts = [r[1] for r in results]
                client_losses = [r[2] for r in results]
                
                # Server aggregation
                avg_loss = np.mean(client_losses)
                if args.aggregation == "fedavg":
                    aggregated_weights = aggregate_fedavg(client_weights, client_sample_counts,
                                                      skip_bn_stats=(args.bn_mode == 'recalib'))
                else:
                    aggregated_weights = aggregate_adaptive_robust(
                        client_weights, client_sample_counts, global_model.state_dict(),
                        eval_model=global_model, val_loader=val_loader_agg, device=device,
                        num_eval_classes=len(seen_classes))
                global_model.load_state_dict(aggregated_weights)
                if args.bn_mode == 'recalib':
                    recalibrate_bn(global_model, val_loader_agg, device)

                print(f"Task {task_idx} [Phase 2] Round {epoch+1}/{args.epochs_novel} => Avg Client Loss: {avg_loss:.4f}")
                
                # Server Evaluation
                test_dataset = dm.get_test_dataset(seen_classes, max_samples_per_class=_test_cap)
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
                metrics_round, y_pred, y_true = evaluate(global_model, test_loader, device)
                
                round_idx = args.epochs_base + (task_idx - 2) * (args.epochs_novel + args.epochs_crt) + epoch + 1
                round_csv_writer.writerow([
                    round_idx, task_idx, 2, epoch + 1, round(avg_loss, 4), round(metrics_round['loss'], 4),
                    round(metrics_round['acc'], 4),
                    round(metrics_round['prec_mic'], 4), round(metrics_round['prec_mac'], 4), round(metrics_round['prec_wei'], 4),
                    round(metrics_round['rec_mic'], 4),  round(metrics_round['rec_mac'], 4),  round(metrics_round['rec_wei'], 4),
                    round(metrics_round['f1_mic'], 4),   round(metrics_round['f1_mac'], 4),   round(metrics_round['f1_wei'], 4),
                    round(metrics_round['fpr'], 4)
                ])
                round_csv_file.flush()
                
                # Save confusion matrix for this round
                cm_dir = os.path.join(run_dir, "confusion_matrices")
                os.makedirs(cm_dir, exist_ok=True)
                plot_confusion_matrix_round(y_true, y_pred, round_idx, task_idx, 2, epoch + 1, cm_dir)
                
                epoch_checkpoint_path = os.path.join(run_dir, "checkpoints", "resume_latest.pt")
                os.makedirs(os.path.dirname(epoch_checkpoint_path), exist_ok=True)
                torch.save({
                    'task_idx': task_idx,
                    'phase': 2,
                    'epoch': epoch + 1,
                    'state_dict': global_model.state_dict(),
                    'checkpoint_init': global_model.checkpoint_init,
                    'seen_classes': seen_classes,
                    'client_memories': client_memories,
                }, epoch_checkpoint_path)

        # ---------------------------------------------------------------------
        # PHASE 3: Calibration without Forgetting (CRT)
        # ---------------------------------------------------------------------
        run_phase3 = (task_idx > start_task) or (start_phase <= 3)
        
        if run_phase3:
            print(f"\n--- [Phase 3 - Federated] Classifier Re-training (Logits Calibration) ---")
            
            # Freeze encoder on global model for Phase 3
            for param in global_model.encoder.parameters():
                param.requires_grad = False
                
            # Pre-create calibration datasets and loaders once (reused across all Phase 3 rounds)
            client_cal_loaders = []
            client_cal_sizes = []
            for c in range(num_clients):
                c_x, c_y = client_datasets[c]
                if c_x is None:
                    client_cal_loaders.append(None)
                    client_cal_sizes.append(0)
                    continue
                    
                cal_dataset = client_get_calibration_dataset(
                    seen_classes=seen_classes,
                    current_task_x=c_x,
                    current_task_y=c_y,
                    current_task_classes=classes,
                    exemplar_memory=client_memories[c]
                )
                if cal_dataset is not None:
                    cal_loader = DataLoader(cal_dataset, batch_size=min(args.batch_size, len(cal_dataset)), shuffle=True)
                    client_cal_loaders.append(cal_loader)
                    client_cal_sizes.append(len(cal_dataset))
                else:
                    client_cal_loaders.append(None)
                    client_cal_sizes.append(0)
            
            initial_epoch = start_epoch if (task_idx == start_task and start_phase == 3) else 0
            
            for epoch in range(initial_epoch, args.epochs_crt):
                # Worker definition for concurrent execution of Phase 3
                def train_local_client_phase3(c):
                    torch.set_num_threads(1)
                    
                    cal_loader = client_cal_loaders[c]
                    cal_size = client_cal_sizes[c]
                    
                    local_model = CICIoT23Model().to(device)
                    local_model.fc = nn.Linear(local_model.encoder.out_dim, global_model.fc.out_features, bias=False).to(device)
                    local_model.load_state_dict(global_model.state_dict())
                    
                    # Freeze local model's encoder
                    for param in local_model.encoder.parameters():
                        param.requires_grad = False
                    local_model.train()
                    
                    optimizer_crt = torch.optim.Adam([local_model.fc.weight], lr=args.lr * 0.5)
                    losses = 0.0
                    for local_ep in range(args.local_epochs):
                        for batch in cal_loader:
                            inputs = batch['data'].to(device)
                            targets = batch['label'].to(device)
                            
                            optimizer_crt.zero_grad()
                            logits = local_model(inputs)
                            seen_logits = logits[:, :len(seen_classes)]
                            loss = F.cross_entropy(seen_logits, targets)
                                
                            loss.backward()
                            optimizer_crt.step()
                            
                            losses += loss.item()
                            
                    avg_l = losses / (len(cal_loader) * args.local_epochs)
                    return local_model.state_dict(), cal_size, avg_l

                # Run parallel training
                active_cal_clients = [c for c in range(num_clients) if client_cal_loaders[c] is not None]
                if len(active_cal_clients) > 0:
                    with ThreadPoolExecutor(max_workers=len(active_cal_clients)) as executor:
                        results = list(executor.map(train_local_client_phase3, active_cal_clients))
                        
                    client_weights = [r[0] for r in results]
                    client_sample_counts = [r[1] for r in results]
                    client_losses = [r[2] for r in results]
                else:
                    client_weights = []
                    client_sample_counts = []
                    client_losses = []
                    
                if len(client_weights) > 0:
                    # Server aggregation
                    avg_loss = np.mean(client_losses)
                    if args.aggregation == "fedavg":
                        aggregated_weights = aggregate_fedavg(client_weights, client_sample_counts,
                                                      skip_bn_stats=(args.bn_mode == 'recalib'))
                    else:
                        aggregated_weights = aggregate_adaptive_robust(
                            client_weights, client_sample_counts, global_model.state_dict(),
                            eval_model=global_model, val_loader=val_loader_agg, device=device,
                            num_eval_classes=len(seen_classes))
                    global_model.load_state_dict(aggregated_weights)
                    if args.bn_mode == 'recalib':
                        recalibrate_bn(global_model, val_loader_agg, device)
                else:
                    avg_loss = 0.0
                    
                print(f"Task {task_idx} [Phase 3 - CRT] Round {epoch+1}/{args.epochs_crt} => Avg Client Loss: {avg_loss:.4f}")
                
                # Server Evaluation
                test_dataset = dm.get_test_dataset(seen_classes, max_samples_per_class=_test_cap)
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
                metrics_round, y_pred, y_true = evaluate(global_model, test_loader, device)
                
                round_idx = args.epochs_base + (task_idx - 2) * (args.epochs_novel + args.epochs_crt) + args.epochs_novel + epoch + 1
                round_csv_writer.writerow([
                    round_idx, task_idx, 3, epoch + 1, round(avg_loss, 4), round(metrics_round['loss'], 4),
                    round(metrics_round['acc'], 4),
                    round(metrics_round['prec_mic'], 4), round(metrics_round['prec_mac'], 4), round(metrics_round['prec_wei'], 4),
                    round(metrics_round['rec_mic'], 4),  round(metrics_round['rec_mac'], 4),  round(metrics_round['rec_wei'], 4),
                    round(metrics_round['f1_mic'], 4),   round(metrics_round['f1_mac'], 4),   round(metrics_round['f1_wei'], 4),
                    round(metrics_round['fpr'], 4)
                ])
                round_csv_file.flush()
                
                # Save confusion matrix for this round
                cm_dir = os.path.join(run_dir, "confusion_matrices")
                os.makedirs(cm_dir, exist_ok=True)
                plot_confusion_matrix_round(y_true, y_pred, round_idx, task_idx, 3, epoch + 1, cm_dir)
                
                epoch_checkpoint_path = os.path.join(run_dir, "checkpoints", "resume_latest.pt")
                os.makedirs(os.path.dirname(epoch_checkpoint_path), exist_ok=True)
                torch.save({
                    'task_idx': task_idx,
                    'phase': 3,
                    'epoch': epoch + 1,
                    'state_dict': global_model.state_dict(),
                    'checkpoint_init': global_model.checkpoint_init,
                    'seen_classes': seen_classes,
                    'client_memories': client_memories,
                }, epoch_checkpoint_path)
                
            # Unfreeze encoder on global model after Phase 3
            for param in global_model.encoder.parameters():
                param.requires_grad = True
                
        # Clients select exemplars locally for current classes
        for c in range(num_clients):
            c_x, c_y = client_datasets[c]
            if c_x is not None:
                client_select_exemplars(client_memories[c], c_x, c_y, classes, m_per_class=args.memory_per_class)
                
        # Server Evaluation
        test_dataset = dm.get_test_dataset(seen_classes, max_samples_per_class=_test_cap)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        metrics, y_pred, y_true = evaluate(global_model, test_loader, device)
        
        print(f"\nTask {task_idx} Final Metrics => Acc: {metrics['acc']:.2f}%, F1-Macro: {metrics['f1_mac']:.2f}%")
        csv_writer.writerow([
            task_idx, round(metrics['acc'], 4),
            round(metrics['prec_mic'], 4), round(metrics['prec_mac'], 4), round(metrics['prec_wei'], 4),
            round(metrics['rec_mic'], 4),  round(metrics['rec_mac'], 4),  round(metrics['rec_wei'], 4),
            round(metrics['f1_mic'], 4),   round(metrics['f1_mac'], 4),   round(metrics['f1_wei'], 4)
        ])
        csv_file.flush()
        
        plot_confusion_matrix(y_true, y_pred, task_idx, run_dir)
        
        task_checkpoint_path = os.path.join(run_dir, f"checkpoint_task_{task_idx}.pt")
        torch.save({
            'task_idx': task_idx,
            'state_dict': global_model.state_dict(),
            'seen_classes': seen_classes,
            'client_memories': client_memories,
            'checkpoint_init': global_model.checkpoint_init,
        }, task_checkpoint_path)
        print(f"[Checkpoint] Saved final task checkpoint to: {task_checkpoint_path}")
        
        start_phase = 2
        
    csv_file.close()
    round_csv_file.close()
    print("\n" + "="*80)
    print(f"RUN COMPLETE. Results saved in: {run_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
