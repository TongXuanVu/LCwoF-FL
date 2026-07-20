"""
Remap label cho bo data 100-client.

Bo data 100-client (`FL/core/data_split/100 client`) GIU NGUYEN label ID goc cua CIC-IoT23
(`preserve_original_label_ids: true`) va co thu tu task PHI TUAN TU, mo ta trong file
`task_mapping_label_ids.json`:
    Task 1: [1, 0, 11, 12, 27, 26]
    Task 2: [2, 14, 25, 24, 20, 28]
    ...
Trong khi code CIL gia dinh label tuan tu: task 1 = [0..5], task 2 = [6..11], ...

Module nay build LUT map label goc -> label tuan tu theo dung thu tu task.
Neu KHONG tim thay file json (bo data cu da tuan tu san) thi khong doi gi -> tuong thich nguoc.
"""
import os
import json
import glob

import torch

_LUT = None
_LOADED = False


def _find_mapping_file(data_root=None):
    candidates = []
    if data_root:
        candidates += [
            os.path.join(data_root, "task_mapping_label_ids.json"),
            os.path.join(data_root, "data", "task_mapping_label_ids.json"),
        ]
    if os.path.exists("/kaggle/input"):
        candidates += sorted(glob.glob("/kaggle/input/**/task_mapping_label_ids.json", recursive=True))
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "task_mapping_label_ids.json"))
    return next((p for p in candidates if p and os.path.exists(p)), None)


def init_remap(data_root=None):
    """Nap LUT mot lan. Tra ve True neu co remap, False neu data da tuan tu."""
    global _LUT, _LOADED
    if _LOADED:
        return _LUT is not None

    _LOADED = True
    path = _find_mapping_file(data_root)
    if path is None:
        print("[LabelRemap] Khong thay task_mapping_label_ids.json -> gia dinh label da tuan tu.")
        return False

    with open(path, "r") as f:
        task_orders = json.load(f)
    flat = [int(c) for task in task_orders for c in task]
    if sorted(flat) != list(range(len(flat))):
        print(f"[LabelRemap] CANH BAO: {path} khong phu kin 0..N-1 -> bo qua remap.")
        return False

    lut = torch.full((max(flat) + 1,), -1, dtype=torch.long)
    for seq_id, orig_id in enumerate(flat):
        lut[orig_id] = seq_id
    _LUT = lut
    print(f"[LabelRemap] Remap label goc -> tuan tu theo: {path}")
    print(f"[LabelRemap] Thu tu task (label goc): {task_orders}")
    return True


def remap(y):
    """Ap LUT cho tensor label y. No-op neu data da tuan tu."""
    if not _LOADED:
        init_remap()
    if _LUT is None or y is None:
        return y
    if not torch.is_tensor(y):
        y = torch.as_tensor(y)
    out = _LUT[y.long()]
    if (out < 0).any():
        bad = torch.unique(y[out < 0]).tolist()
        raise ValueError(f"[LabelRemap] Label {bad} khong co trong task_mapping_label_ids.json")
    return out
