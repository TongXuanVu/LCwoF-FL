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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


class CICIoT23Model(nn.Module):
    """
    CNN1D + Dynamic Fully Connected Layer model for Class-Incremental Learning on CIC-IoT23.
    """
    def __init__(self):
        super(CICIoT23Model, self).__init__()
        self.encoder = CNN1DConvNet()
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
            
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
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
    
    # Expand/resize model.fc to match num_classes
    model.fc = nn.Linear(model.encoder.out_dim, num_classes, bias=False)
    model.load_state_dict(state_dict)
    
    # Restore checkpoint_init (L2 regularization anchor) and rebuild cached parameter pairs
    if 'checkpoint_init' in checkpoint and checkpoint['checkpoint_init'] is not None:
        model.checkpoint_init = checkpoint['checkpoint_init']
        model.l2_params = []
        for name, param in model.encoder.named_parameters():
            if name in model.checkpoint_init:
                model.l2_params.append((param, model.checkpoint_init[name]))
        
    return checkpoint


def aggregate_fedavg(client_weights, client_sample_counts):
    total_samples = sum(client_sample_counts)
    if total_samples == 0:
        return client_weights[0]
    
    w_avg = copy.deepcopy(client_weights[0])
    for key in w_avg.keys():
        w_avg[key] = torch.zeros_like(w_avg[key])
        
    for i in range(len(client_weights)):
        weight = client_sample_counts[i] / total_samples
        for key in w_avg.keys():
            if torch.is_floating_point(client_weights[i][key]):
                w_avg[key] += client_weights[i][key] * weight
            else:
                if i == 0:
                    w_avg[key] = client_weights[i][key]
                else:
                    w_avg[key] = torch.max(w_avg[key], client_weights[i][key])
    return w_avg


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
                cal_x_list.append(exemplar_memory[c]['x'])
                cal_y_list.append(exemplar_memory[c]['y'])

    # 2. Current classes from local training data
    for c in current_task_classes:
        class_mask = (current_task_y == c)
        x_c = current_task_x[class_mask]
        y_c = current_task_y[class_mask]
        
        n_samples = x_c.shape[0]
        if n_samples > 0:
            n_select = max(1, int(n_samples * 0.01))
            indices = np.random.choice(n_samples, n_select, replace=False)
            cal_x_list.append(x_c[indices])
            cal_y_list.append(y_c[indices])

    if len(cal_x_list) == 0:
        return None

    cal_x = torch.cat(cal_x_list, dim=0)
    cal_y = torch.cat(cal_y_list, dim=0)
    return CICIoT23Dataset(cal_x, cal_y)


def main():
    parser = argparse.ArgumentParser(description="LCwoF training in Federated Learning scenario")
    parser.add_argument("--data_root", type=str, default="C:/FederatedLearning/FL/core/data_split",
                        help="Path to the directory containing global_test_data.pt and federated_data")
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
    parser.add_argument("--max_test_samples_per_class", type=int, default=5000,
                        help="Maximum test samples per class to evaluate on (default 5000)")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test", "resume"],
                        help="Execution mode: train (default), test (evaluate checkpoints), or resume (resume training)")
    parser.add_argument("--resume_path", type=str, default="",
                        help="Path to a model checkpoint (.pt) to resume from or to test")
    parser.add_argument("--test_dir", type=str, default="",
                        help="Path to directory containing multiple checkpoints to evaluate in test mode")
    parser.add_argument("--run_dir", type=str, default="",
                        help="Custom directory to save logs, CSV, and checkpoints")

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
    client_memories = [{} for _ in range(10)]
    
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
                    
                test_dataset = dm.get_test_dataset(seen_classes, max_samples_per_class=args.max_test_samples_per_class)
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
            print(f"[DataManager] Restored exemplar memory containing {len(client_memories)} clients.")
            
        filename = os.path.basename(args.resume_path)
        if "checkpoint_task" in filename:
            task_idx_completed = checkpoint.get('task_idx', 1)
            start_task = task_idx_completed + 1
            start_epoch = 0
            start_phase = 2 if start_task > 1 else 1
            print(f"[Resume] Final Task {task_idx_completed} checkpoint detected. Resuming from Task {start_task} (Pha {start_phase}, Round 1).")
        elif "round" in filename or "epoch" in filename:
            start_task = checkpoint.get('task_idx', 1)
            start_epoch = checkpoint.get('epoch', 0)
            start_phase = checkpoint.get('phase', 1)
            print(f"[Resume] Round checkpoint detected. Resuming Task {start_task} (Pha {start_phase}) starting at Round {start_epoch + 1}.")
            
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
            
        num_clients = 5 # 5 clients for Task 1
        print("\n" + "="*80)
        print(f"STARTING TASK 1 - FEDERATED (Base classes {classes}, Clients: {num_clients})")
        print("="*80)
        
        # Load local client datasets
        client_datasets = []
        for c in range(num_clients):
            c_path = os.path.join(args.data_root, "federated_data", f"client_{c}_task_1.pt")
            c_data = torch.load(c_path, map_location="cpu", weights_only=False)
            c_x, c_y = c_data["x"].float(), c_data["y"].long()
            if args.debug:
                c_x = c_x[:500]
                c_y = c_y[:500]
            client_datasets.append((c_x, c_y))
            print(f"  Client {c} loaded: {len(c_x)} training samples.")
            
        # Pre-create DataLoaders for all clients to avoid creation overhead in round loop
        client_loaders = []
        for c in range(num_clients):
            c_x, c_y = client_datasets[c]
            train_dataset = CICIoT23Dataset(c_x, c_y)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            client_loaders.append(train_loader)
            
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
                
                # Optional AMP support
                scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
                
                losses = 0.0
                for local_ep in range(args.local_epochs):
                    for batch in train_loader:
                        inputs = batch['data'].to(device)
                        targets = batch['label'].to(device)
                        
                        optimizer.zero_grad()
                        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                            logits = local_model(inputs)
                            loss = F.cross_entropy(logits, targets)
                            
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        
                        losses += loss.item()
                        
                avg_l = losses / (len(train_loader) * args.local_epochs)
                return local_model.state_dict(), len(c_x), avg_l

            # Execute client training concurrently
            with ThreadPoolExecutor(max_workers=num_clients) as executor:
                results = list(executor.map(train_local_client_task1, range(num_clients)))
                
            client_weights = [r[0] for r in results]
            client_sample_counts = [r[1] for r in results]
            client_losses = [r[2] for r in results]
                
            # Server FedAvg aggregation
            avg_loss = np.mean(client_losses)
            aggregated_weights = aggregate_fedavg(client_weights, client_sample_counts)
            global_model.load_state_dict(aggregated_weights)
            
            print(f"Task 1 Round {epoch+1}/{args.epochs_base} => Avg Client Loss: {avg_loss:.4f}")
            
            # Server Evaluation
            test_dataset = dm.get_test_dataset(seen_classes, max_samples_per_class=args.max_test_samples_per_class)
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
            epoch_checkpoint_path = os.path.join(run_dir, "checkpoints", f"task_1_round_{epoch+1}.pt")
            os.makedirs(os.path.dirname(epoch_checkpoint_path), exist_ok=True)
            torch.save({
                'task_idx': 1,
                'epoch': epoch + 1,
                'state_dict': global_model.state_dict(),
                'checkpoint_init': global_model.checkpoint_init,
                'seen_classes': seen_classes,
                'client_memories': client_memories,
            }, epoch_checkpoint_path)
            
        # Clients select exemplars locally
        for c in range(num_clients):
            c_x, c_y = client_datasets[c]
            client_select_exemplars(client_memories[c], c_x, c_y, classes, m_per_class=args.memory_per_class)
            
        global_model.save_base_checkpoint()
        
        test_dataset = dm.get_test_dataset(seen_classes, max_samples_per_class=args.max_test_samples_per_class)
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
                
        num_clients = 4 + task_idx  # Task 2: 6, Task 3: 7, ..., Task 6: 10 clients
        print("\n" + "="*80)
        print(f"STARTING TASK {task_idx} - FEDERATED (Novel classes {classes}, Clients: {num_clients})")
        print("="*80)
        
        # Load local client datasets
        client_datasets = []
        for c in range(num_clients):
            c_path = os.path.join(args.data_root, "federated_data", f"client_{c}_task_{task_idx}.pt")
            if os.path.exists(c_path):
                c_data = torch.load(c_path, map_location="cpu", weights_only=False)
                c_x, c_y = c_data["x"].float(), c_data["y"].long()
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
                                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
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
                    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
                    
                    losses = 0.0
                    for local_ep in range(args.local_epochs):
                        for batch in train_loader:
                            inputs = batch['data'].to(device)
                            targets = batch['label'].to(device)
                            
                            optimizer.zero_grad()
                            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                                logits = local_model(inputs)
                                local_logits = logits[:, classes]
                                local_targets = torch.tensor([classes.index(t.item()) for t in targets], dtype=torch.long, device=device)
                                
                                cls_loss = F.cross_entropy(local_logits, local_targets)
                                l2_loss = local_model.L2_weight_loss(l2_lambda=args.l2_lambda)
                                total_loss = cls_loss + l2_loss
                                
                            scaler.scale(total_loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                            
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
                aggregated_weights = aggregate_fedavg(client_weights, client_sample_counts)
                global_model.load_state_dict(aggregated_weights)
                
                print(f"Task {task_idx} [Phase 2] Round {epoch+1}/{args.epochs_novel} => Avg Client Loss: {avg_loss:.4f}")
                
                # Server Evaluation
                test_dataset = dm.get_test_dataset(seen_classes, max_samples_per_class=args.max_test_samples_per_class)
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
                
                epoch_checkpoint_path = os.path.join(run_dir, "checkpoints", f"task_{task_idx}_phase2_round_{epoch+1}.pt")
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
                    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
                    
                    losses = 0.0
                    for local_ep in range(args.local_epochs):
                        for batch in cal_loader:
                            inputs = batch['data'].to(device)
                            targets = batch['label'].to(device)
                            
                            optimizer_crt.zero_grad()
                            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                                logits = local_model(inputs)
                                seen_logits = logits[:, :len(seen_classes)]
                                loss = F.cross_entropy(seen_logits, targets)
                                
                            scaler.scale(loss).backward()
                            scaler.step(optimizer_crt)
                            scaler.update()
                            
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
                    aggregated_weights = aggregate_fedavg(client_weights, client_sample_counts)
                    global_model.load_state_dict(aggregated_weights)
                else:
                    avg_loss = 0.0
                    
                print(f"Task {task_idx} [Phase 3 - CRT] Round {epoch+1}/{args.epochs_crt} => Avg Client Loss: {avg_loss:.4f}")
                
                # Server Evaluation
                test_dataset = dm.get_test_dataset(seen_classes, max_samples_per_class=args.max_test_samples_per_class)
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
                
                epoch_checkpoint_path = os.path.join(run_dir, "checkpoints", f"task_{task_idx}_phase3_round_{epoch+1}.pt")
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
        test_dataset = dm.get_test_dataset(seen_classes, max_samples_per_class=args.max_test_samples_per_class)
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
