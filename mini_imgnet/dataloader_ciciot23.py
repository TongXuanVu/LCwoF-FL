import os
import torch
import numpy as np
from torch.utils.data import Dataset

class CICIoT23Dataset(Dataset):
    """
    Dataset wrapper for tabular CIC-IoT23 data returning dict {'data': tensor, 'label': tensor}.
    """
    def __init__(self, x, y):
        # Convert to tensors
        if isinstance(x, np.ndarray):
            self.x = torch.from_numpy(x).float()
        elif isinstance(x, torch.Tensor):
            self.x = x.float()
        else:
            self.x = torch.tensor(x, dtype=torch.float32)

        if isinstance(y, np.ndarray):
            self.y = torch.from_numpy(y).long()
        elif isinstance(y, torch.Tensor):
            self.y = y.long()
        else:
            self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {
            'data': self.x[idx],
            'label': self.y[idx]
        }


class CICIoT23DataManager:
    """
    Manages loading of centralized CIC-IoT23 tasks and global test data.
    """
    def __init__(self, data_root="C:/FederatedLearning/FL/core/data_split", use_fewshot=False, use_10shot=False, data_dir_name=None):
        self.data_root = data_root
        if data_dir_name:
            self.centralized_dir = os.path.join(data_root, data_dir_name)
        elif use_10shot:
            self.centralized_dir = os.path.join(data_root, "10shot", "centralized_data_10shot")
        elif use_fewshot:
            self.centralized_dir = os.path.join(data_root, "fewshot", "centralized_data_fewshot")
        else:
            self.centralized_dir = os.path.join(data_root, "centralized_data")
        self.global_test_file = os.path.join(data_root, "global_test_data.pt")
        if not os.path.exists(self.global_test_file):
            self.global_test_file = os.path.join(data_root, "data", "global_test_data.pt")
        
        # Verify paths
        if not os.path.exists(self.centralized_dir):
            raise FileNotFoundError(f"Centralized data directory not found: {self.centralized_dir}")
        if not os.path.exists(self.global_test_file):
            raise FileNotFoundError(f"Global test file not found: {self.global_test_file} or {os.path.join(data_root, 'global_test_data.pt')}")

        # Load global test data
        print("[DataManager] Loading global test data...")
        test_dict = torch.load(self.global_test_file, map_location="cpu", weights_only=False)
        self.test_x = test_dict["x"].float()
        self.test_y = test_dict["y"].long()
        print(f"[DataManager] Loaded test set: {self.test_x.shape[0]} samples")

        # Memory for exemplars: class_idx -> {'x': tensor, 'y': tensor}
        self.exemplar_memory = {}

    def load_task_train_data(self, task_id):
        """
        Loads training data for a specific task (1-indexed: 1 to 6).
        Returns x, y tensors.
        """
        path = os.path.join(self.centralized_dir, f"centralized_task_{task_id}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Task data file not found: {path}")
        
        data = torch.load(path, map_location="cpu", weights_only=False)
        return data["x"].float(), data["y"].long()

    def get_test_dataset(self, seen_classes, max_samples_per_class=None):
        """
        Returns a CICIoT23Dataset containing test samples belonging to seen_classes.
        Optional max_samples_per_class limits test set size for faster evaluation.
        """
        x_filtered_list = []
        y_filtered_list = []
        
        for c in seen_classes:
            class_mask = (self.test_y == c)
            x_c = self.test_x[class_mask]
            y_c = self.test_y[class_mask]
            
            if len(x_c) > 0:
                if max_samples_per_class is not None and len(x_c) > max_samples_per_class:
                    # Random sample
                    indices = np.random.choice(len(x_c), max_samples_per_class, replace=False)
                    x_filtered_list.append(x_c[indices])
                    y_filtered_list.append(y_c[indices])
                else:
                    x_filtered_list.append(x_c)
                    y_filtered_list.append(y_c)
                    
        if len(x_filtered_list) == 0:
            return CICIoT23Dataset(torch.empty(0, self.test_x.shape[1]), torch.empty(0, dtype=torch.long))
            
        x_filtered = torch.cat(x_filtered_list, dim=0)
        y_filtered = torch.cat(y_filtered_list, dim=0)
        return CICIoT23Dataset(x_filtered, y_filtered)

    def select_exemplars(self, x, y, classes, m_per_class=20):
        """
        Selects m_per_class samples for each class in the current task and saves them to exemplar_memory.
        """
        print(f"[DataManager] Selecting {m_per_class} exemplars per class for {classes}...")
        for c in classes:
            class_mask = (y == c)
            x_c = x[class_mask]
            y_c = y[class_mask]
            
            n_samples = x_c.shape[0]
            if n_samples == 0:
                print(f"[DataManager] Warning: No samples found for class {c}!")
                continue
                
            # Randomly select exemplars (1% of data)
            n_select = max(1, int(n_samples * 0.01))
            indices = np.random.choice(n_samples, n_select, replace=False)
            self.exemplar_memory[c] = {
                'x': x_c[indices],
                'y': y_c[indices]
            }
        print("[DataManager] Exemplar memory updated.")

    def get_calibration_dataset(self, seen_classes, current_task_x, current_task_y, current_task_classes, m_per_class=20):
        """
        Constructs a perfectly balanced calibration dataset (Phase 3 - CRT):
        - For old classes: loads from exemplar_memory.
        - For current task classes: samples m_per_class from the current training data.
        """
        cal_x_list = []
        cal_y_list = []

        # 1. Old classes from memory
        for c in seen_classes:
            if c not in current_task_classes:
                if c in self.exemplar_memory:
                    cal_x_list.append(self.exemplar_memory[c]['x'])
                    cal_y_list.append(self.exemplar_memory[c]['y'])
                else:
                    print(f"[DataManager] Warning: Class {c} not found in exemplar memory!")

        # 2. Current classes from current task data
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
            raise ValueError("No calibration samples could be loaded!")

        cal_x = torch.cat(cal_x_list, dim=0)
        cal_y = torch.cat(cal_y_list, dim=0)
        
        print(f"[DataManager] Constructed balanced calibration dataset with {cal_x.shape[0]} samples.")
        return CICIoT23Dataset(cal_x, cal_y)
