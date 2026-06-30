import os
import sys
import time
import csv
import argparse
import random
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
        print("[Model] Base checkpoint saved for L2 parameter regularization.")

    def L2_weight_loss(self, l2_lambda=500.0):
        """
        L2 Parameter Regularization (Anti-forgetting loss)
        Formula: L2 = l2_lambda * sum((W_base - W_current)^2)
        """
        if self.checkpoint_init is None:
            return 0.0
            
        loss = 0.0
        for name, param in self.encoder.named_parameters():
            if name in self.checkpoint_init:
                loss += (self.checkpoint_init[name] - param).pow(2).sum()
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
    all_preds = []
    all_targets = []
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['data'].to(device)
            targets = batch['label'].to(device)
            
            logits = model(inputs)
            
            # Compute loss
            loss = F.cross_entropy(logits, targets)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    
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
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    
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
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'LCwoF Confusion Matrix - Task {task_id}')
    
    save_path = os.path.join(run_dir, f'confusion_matrix_task_{task_id}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved confusion matrix to: {save_path}")


def plot_confusion_matrix_round(y_true, y_pred, round_idx, task, phase, epoch, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'LCwoF Confusion Matrix - Round {round_idx} (Task {task}, Phase {phase}, Epoch {epoch})')
    
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
    
    # Restore checkpoint_init (L2 regularization anchor)
    if 'checkpoint_init' in checkpoint:
        model.checkpoint_init = checkpoint['checkpoint_init']
        
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description="LCwoF training on Centralized CIC-IoT23 dataset")
    parser.add_argument("--data_root", type=str, default="C:/FederatedLearning/FL/core/data_split",
                        help="Path to the directory containing global_test_data.pt and centralized_data")
    parser.add_argument("--l2_lambda", type=float, default=500.0,
                        help="Lambda coefficient for L2 parameter regularization (default 500.0)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default 0.001)")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size (default 1024)")
    parser.add_argument("--epochs_base", type=int, default=30, help="Training epochs for base task (default 30)")
    parser.add_argument("--epochs_novel", type=int, default=30, help="Training epochs for novel tasks (default 30)")
    parser.add_argument("--epochs_crt", type=int, default=30, help="Calibration training epochs (default 30)")
    parser.add_argument("--memory_per_class", type=int, default=20, help="Exemplar memory budget per class (default 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    parser.add_argument("--debug", action="store_true", help="Debug mode: fast training with 2 epochs")
    parser.add_argument("--max_test_samples_per_class", type=int, default=5000,
                        help="Maximum test samples per class to evaluate on (default 5000 to keep CPU evaluation fast)")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test", "resume"],
                        help="Execution mode: train (default), test (evaluate checkpoints), or resume (resume training)")
    parser.add_argument("--resume_path", type=str, default="",
                        help="Path to a model checkpoint (.pt) to resume from or to test")
    parser.add_argument("--test_dir", type=str, default="",
                        help="Path to directory containing multiple checkpoints to evaluate in test mode")
    parser.add_argument("--run_dir", type=str, default="",
                        help="Custom directory to save logs, CSV, and checkpoints")
    
    parser.add_argument("--use_fewshot", action="store_true",
                        help="Use the few-shot data split (centralized_data_fewshot) instead of the full centralized data")
    parser.add_argument("--use_10shot", action="store_true",
                        help="Use the 10-shot data split (10shot/centralized_data_10shot) instead of the full centralized data")
    parser.add_argument("--data_dir_name", type=str, default="",
                        help="Custom directory name for centralized data (e.g., 'centralized_data_10shot'). Overrides other flags if provided.")
    
    args = parser.parse_args()
    
    if args.debug:
        print("[DEBUG] Debug mode activated! Adjusting epochs to 2 and test sampling to 200 for quick verification.")
        args.epochs_base = 2
        args.epochs_novel = 2
        args.epochs_crt = 2
        args.max_test_samples_per_class = 200
        
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using device: {device}")
    
    # 1. Setup Data Manager
    dm = CICIoT23DataManager(data_root=args.data_root, use_fewshot=args.use_fewshot, use_10shot=args.use_10shot, data_dir_name=args.data_dir_name)
    
    # Define tasks and their corresponding classes
    tasks_classes = [
        [0, 1, 2, 3, 4, 5],       # Task 1 (Base task)
        [6, 7, 8, 9, 10, 11],     # Task 2
        [12, 13, 14, 15, 16, 17], # Task 3
        [18, 19, 20, 21, 22, 23], # Task 4
        [24, 25, 26, 27, 28],     # Task 5
        [29, 30, 31, 32, 33]      # Task 6
    ]
    
    # Initialize Model
    model = CICIoT23Model().to(device)
    
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
                elif "epoch" in name:
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
                checkpoint = load_checkpoint(model, cp_path, device)
                if 'seen_classes' in checkpoint:
                    seen_classes = checkpoint['seen_classes']
                else:
                    num_classes = model.fc.out_features
                    seen_classes = []
                    for tc in tasks_classes:
                        if len(seen_classes) >= num_classes:
                            break
                        seen_classes.extend(tc)
                    seen_classes = seen_classes[:num_classes]
                    
                test_dataset = dm.get_test_dataset(seen_classes, max_samples_per_class=args.max_test_samples_per_class)
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
                metrics, _, _ = evaluate(model, test_loader, device)
                
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
        
        checkpoint = load_checkpoint(model, args.resume_path, device)
        
        if 'seen_classes' in checkpoint:
            seen_classes = checkpoint['seen_classes']
        else:
            num_classes = model.fc.out_features
            seen_classes = []
            for tc in tasks_classes:
                if len(seen_classes) >= num_classes:
                    break
                seen_classes.extend(tc)
            seen_classes = seen_classes[:num_classes]
            
        if 'exemplar_memory' in checkpoint:
            dm.exemplar_memory = checkpoint['exemplar_memory']
            print(f"[DataManager] Restored exemplar memory containing {len(dm.exemplar_memory.keys())} classes.")
            
        filename = os.path.basename(args.resume_path)
        if "checkpoint_task" in filename:
            task_idx_completed = checkpoint.get('task_idx', 1)
            start_task = task_idx_completed + 1
            start_epoch = 0
            start_phase = 2 if start_task > 1 else 1
            print(f"[Resume] Final Task {task_idx_completed} checkpoint detected. Resuming from Task {start_task} (Pha {start_phase}, Epoch 1).")
        elif "epoch" in filename:
            start_task = checkpoint.get('task_idx', 1)
            start_epoch = checkpoint.get('epoch', 0)
            start_phase = checkpoint.get('phase', 1)
            print(f"[Resume] Epoch checkpoint detected. Resuming Task {start_task} (Pha {start_phase}) starting at Epoch {start_epoch + 1}.")
            
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
            run_dir = os.path.join(str(dir_path), "logs", "lcwof_ciciot23", f"seed{args.seed}_{timestamp}")
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
            "round_idx", "task", "phase", "epoch", "train_loss", "test_loss",
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
        
        print("\n" + "="*80)
        print(f"STARTING TASK 1 (Base classes {classes})")
        print("="*80)
        
        train_x, train_y = dm.load_task_train_data(task_id=1)
        
        if args.debug:
            train_x = train_x[:5000]
            train_y = train_y[:5000]
            
        train_dataset = CICIoT23Dataset(train_x, train_y)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        
        initial_epoch = start_epoch if (start_task == 1) else 0
        
        for epoch in range(initial_epoch, args.epochs_base):
            model.train()
            losses = 0.0
            correct, total = 0, 0
            
            for batch in train_loader:
                inputs = batch['data'].to(device)
                targets = batch['label'].to(device)
                
                logits = model(inputs)
                loss = F.cross_entropy(logits, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets).cpu().sum().item()
                total += targets.size(0)
                
            print(f"Task 1 Epoch {epoch+1}/{args.epochs_base} => Loss: {losses/len(train_loader):.4f}, Train Acc: {correct*100/total:.2f}%")
            
            # Evaluate after this round
            test_dataset = dm.get_test_dataset(seen_classes, max_samples_per_class=args.max_test_samples_per_class)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            metrics_round, y_pred, y_true = evaluate(model, test_loader, device)
            
            round_idx = epoch + 1
            round_csv_writer.writerow([
                round_idx, 1, 1, epoch + 1, round(losses/len(train_loader), 4), round(metrics_round['loss'], 4),
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
            epoch_checkpoint_path = os.path.join(run_dir, "checkpoints", f"task_1_epoch_{epoch+1}.pt")
            os.makedirs(os.path.dirname(epoch_checkpoint_path), exist_ok=True)
            torch.save({
                'task_idx': 1,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'checkpoint_init': model.checkpoint_init,
                'seen_classes': seen_classes,
                'exemplar_memory': dm.exemplar_memory,
            }, epoch_checkpoint_path)
            
        dm.select_exemplars(train_x, train_y, classes, m_per_class=args.memory_per_class)
        model.save_base_checkpoint()
        
        test_dataset = dm.get_test_dataset(seen_classes, max_samples_per_class=args.max_test_samples_per_class)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        metrics, y_pred, y_true = evaluate(model, test_loader, device)
        
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
            'state_dict': model.state_dict(),
            'seen_classes': seen_classes,
            'exemplar_memory': dm.exemplar_memory,
            'checkpoint_init': model.checkpoint_init,
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
            
        print("\n" + "="*80)
        print(f"STARTING TASK {task_idx} (Novel classes {classes})")
        print("="*80)
        
        train_x, train_y = dm.load_task_train_data(task_id=task_idx)
        if args.debug:
            train_x = train_x[:5000]
            train_y = train_y[:5000]
            
        expected_classes = len(seen_classes)
        if model.fc.out_features < expected_classes:
            model.expand_classifier(len(classes))
            model.to(device)
            
            print("[Model] Initializing new classifier weights using mean feature prototypes...")
            model.eval()
            with torch.no_grad():
                for c in classes:
                    c_mask = (train_y == c)
                    x_c = train_x[c_mask]
                    if len(x_c) > 0:
                        # Process in batches to prevent CUDA Out Of Memory (OOM)
                        sum_features = torch.zeros(model.encoder.out_dim, device=device)
                        total_c = len(x_c)
                        eval_batch_size = 8192
                        for i in range(0, total_c, eval_batch_size):
                            batch_x = x_c[i:i + eval_batch_size].to(device)
                            features = model.encoder(batch_x)
                            sum_features += features.sum(dim=0)
                        mean_feature = sum_features / total_c
                        mean_feature = F.normalize(mean_feature, p=2, dim=0)
                        model.fc.weight.data[c] = mean_feature
                        
        # ---------------------------------------------------------------------
        # PHASE 2: Explicit learning of novel classes (with L2 parameter loss)
        # ---------------------------------------------------------------------
        run_phase2 = (task_idx > start_task) or (start_phase <= 2)
        
        if run_phase2:
            print(f"\n--- [Phase 2] Training novel classes for Task {task_idx} ---")
            train_dataset = CICIoT23Dataset(train_x, train_y)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), 
                lr=args.lr, 
                weight_decay=1e-4
            )
            
            initial_epoch = start_epoch if (task_idx == start_task and start_phase == 2) else 0
            
            for epoch in range(initial_epoch, args.epochs_novel):
                model.train()
                losses = 0.0
                correct, total = 0, 0
                
                for batch in train_loader:
                    inputs = batch['data'].to(device)
                    targets = batch['label'].to(device)
                    
                    logits = model(inputs)
                    
                    local_logits = logits[:, classes]
                    local_targets = torch.tensor([classes.index(t.item()) for t in targets], dtype=torch.long, device=device)
                    
                    cls_loss = F.cross_entropy(local_logits, local_targets)
                    l2_loss = model.L2_weight_loss(l2_lambda=args.l2_lambda)
                    total_loss = cls_loss + l2_loss
                    
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    
                    losses += total_loss.item()
                    _, preds = torch.max(local_logits, dim=1)
                    correct += preds.eq(local_targets).cpu().sum().item()
                    total += targets.size(0)
                    
                print(f"Task {task_idx} [Phase 2] Epoch {epoch+1}/{args.epochs_novel} => Loss: {losses/len(train_loader):.4f}, Train Acc: {correct*100/total:.2f}%")
                
                # Evaluate after this round
                test_dataset = dm.get_test_dataset(seen_classes, max_samples_per_class=args.max_test_samples_per_class)
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
                metrics_round, y_pred, y_true = evaluate(model, test_loader, device)
                
                round_idx = args.epochs_base + (task_idx - 2) * (args.epochs_novel + args.epochs_crt) + epoch + 1
                round_csv_writer.writerow([
                    round_idx, task_idx, 2, epoch + 1, round(losses/len(train_loader), 4), round(metrics_round['loss'], 4),
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
                
                epoch_checkpoint_path = os.path.join(run_dir, "checkpoints", f"task_{task_idx}_phase2_epoch_{epoch+1}.pt")
                os.makedirs(os.path.dirname(epoch_checkpoint_path), exist_ok=True)
                torch.save({
                    'task_idx': task_idx,
                    'phase': 2,
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'checkpoint_init': model.checkpoint_init,
                    'seen_classes': seen_classes,
                    'exemplar_memory': dm.exemplar_memory,
                }, epoch_checkpoint_path)
                
        # ---------------------------------------------------------------------
        # PHASE 3: Calibration without Forgetting (CRT)
        # ---------------------------------------------------------------------
        run_phase3 = (task_idx > start_task) or (start_phase <= 3)
        
        if run_phase3:
            print(f"\n--- [Phase 3] Classifier Re-training (Logits Calibration) ---")
            for param in model.encoder.parameters():
                param.requires_grad = False
                
            cal_dataset = dm.get_calibration_dataset(
                seen_classes=seen_classes,
                current_task_x=train_x,
                current_task_y=train_y,
                current_task_classes=classes,
                m_per_class=args.memory_per_class
            )
            cal_loader = DataLoader(cal_dataset, batch_size=min(args.batch_size, len(cal_dataset)), shuffle=True)
            
            optimizer_crt = torch.optim.Adam([model.fc.weight], lr=args.lr * 0.5)
            
            initial_epoch = start_epoch if (task_idx == start_task and start_phase == 3) else 0
            
            for epoch in range(initial_epoch, args.epochs_crt):
                model.train()
                losses = 0.0
                correct, total = 0, 0
                
                for batch in cal_loader:
                    inputs = batch['data'].to(device)
                    targets = batch['label'].to(device)
                    
                    logits = model(inputs)
                    seen_logits = logits[:, :len(seen_classes)]
                    
                    loss = F.cross_entropy(seen_logits, targets)
                    
                    optimizer_crt.zero_grad()
                    loss.backward()
                    optimizer_crt.step()
                    
                    losses += loss.item()
                    _, preds = torch.max(seen_logits, dim=1)
                    correct += preds.eq(targets).cpu().sum().item()
                    total += targets.size(0)
                    
                print(f"Task {task_idx} [Phase 3 - CRT] Epoch {epoch+1}/{args.epochs_crt} => Loss: {losses/len(cal_loader):.4f}, Train Acc: {correct*100/total:.2f}%")
                
                # Evaluate after this round
                test_dataset = dm.get_test_dataset(seen_classes, max_samples_per_class=args.max_test_samples_per_class)
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
                metrics_round, y_pred, y_true = evaluate(model, test_loader, device)
                
                round_idx = args.epochs_base + (task_idx - 2) * (args.epochs_novel + args.epochs_crt) + args.epochs_novel + epoch + 1
                round_csv_writer.writerow([
                    round_idx, task_idx, 3, epoch + 1, round(losses/len(cal_loader), 4), round(metrics_round['loss'], 4),
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
                
                epoch_checkpoint_path = os.path.join(run_dir, "checkpoints", f"task_{task_idx}_phase3_epoch_{epoch+1}.pt")
                os.makedirs(os.path.dirname(epoch_checkpoint_path), exist_ok=True)
                torch.save({
                    'task_idx': task_idx,
                    'phase': 3,
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'checkpoint_init': model.checkpoint_init,
                    'seen_classes': seen_classes,
                    'exemplar_memory': dm.exemplar_memory,
                }, epoch_checkpoint_path)
                
            for param in model.encoder.parameters():
                param.requires_grad = True
                
        dm.select_exemplars(train_x, train_y, classes, m_per_class=args.memory_per_class)
        
        test_dataset = dm.get_test_dataset(seen_classes, max_samples_per_class=args.max_test_samples_per_class)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        metrics, y_pred, y_true = evaluate(model, test_loader, device)
        
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
            'state_dict': model.state_dict(),
            'seen_classes': seen_classes,
            'exemplar_memory': dm.exemplar_memory,
            'checkpoint_init': model.checkpoint_init,
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
