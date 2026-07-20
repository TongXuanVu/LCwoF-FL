# LCwoF & LCwoF-FL — Brief (Chat 1)

## Mục tiêu
Low-shot Classification without Forgetting (LCwoF, ICCV'21) áp cho IDS.
- `LCwoF/` = bản centralized (upper-bound để đối chiếu).
- `LCwoF-FL/` = bản federated do mình phát triển lên, dùng aggregation robust kiểu AFSIC-IDS.

## Dataset
CIC-IoT23, 34 lớp chia 6 task (base 6 lớp + 5 task tăng dần).
Data ở `FL/core/data_split/` (`federated_data/client_*_task_*.pt`, `global_test_data.pt`).

## Kiến trúc (giữ nguyên, không đổi)
CNN1D (`mini_imgnet/cnn1d.py`, out_dim=64, có **BatchNorm1d**) + `nn.Linear(bias=False)`.
3 phase mỗi task: (1) base/novel train, (2) L2-anchor novel learning, (3) CRT calibration.

## Entry & lệnh chạy
- Centralized: `python mini_imgnet/run_ciciot23.py --mode train`
- Federated:  `python mini_imgnet/run_ciciot23_fl.py --aggregation robust`
- So sánh: chạy thêm `--aggregation fedavg`, đối chiếu `metrics.csv`.
- Smoke test: thêm `--debug` (2 round).

## Trạng thái hiện tại
- Đã **viết lại `aggregate_adaptive_robust`** trong `run_ciciot23_fl.py`: copy **1:1** hai hàm
  `is_aggregated_state_key` + `compute_aggregation_weights` từ `AFSIC-IDS/utils/aggregation.py`,
  và bước áp alpha copy từ `AFSIC-IDS/trainer.py`. `aggregate_backbone=True` (aggregate cả mạng).
- LCwoF không có prototype memory → `proto_cons=1.0`, `novelty=0.5` (fallback AFSIC, triệt tiêu trong softmax).
  Q hiệu dụng = `1.0·acc − 0.5·drift − 0.2·update_norm`, `acc` = val accuracy server đo per-client.
- Bước train ở client **không đổi**, đã đối chiếu khớp LCwoF gốc (base/L2-novel/CRT).

## Kết quả chạy Kaggle (robust vs fedavg) — ĐÃ CHỐT: robust THUA
| Task | Robust (AFSIC 1:1) Acc/F1 | FedAvg Acc/F1 |
|---|---|---|
| 1 | **28.21% / 15.71%** | **68.77% / 40.67%** |
| 2 | 14.90% / 7.43% | 36.99% / 13.59% |
| 3 | 40.80% / 18.67% | 30.11% / 14.79% |
| 4–6 (robust) | 38.29/22.68 · 31.20/17.16 · 39.32/24.87 | FedAvg timeout ở Task 3 |

**Kết luận:** robust copy 1:1 từ AFSIC **thua FedAvg** (Task 1: 28% vs 69%). Đã quyết định **KHÔNG chạy tiếp** bản robust này.

**Nguyên nhân (từ log):** số hạng phạt drift (`β_drift=0.5 + β_update=0.2`) trong `Q` = "phạt client giàu dữ liệu".
Client 3 (3.15M mẫu, acc tốt nhất) bị bộ lọc MAD loại mọi round (drift ~40 >> client nhỏ ~0);
alpha dồn vào 2 client nhỏ nhất (4k+11k mẫu) → global gần như không thấy dữ liệu → 28%.
LCwoF train **cả mạng** + số bước client lệch ~800 lần → drift ∝ khối lượng dữ liệu, nên penalty của AFSIC
(vốn thiết kế cho backbone đóng băng + adapter nhỏ, drift đồng đều) phản tác dụng.

**Nếu sau này muốn thử lại (chỉ aggregation):** biến thể "Acc-only" — `beta_drift=0, beta_update=0, robust_filter_updates=False`
→ `Q≈acc`, weight theo accuracy. Nhưng khi đó KHÔNG còn 1:1 AFSIC. Gốc rễ (client drift) phải sửa ở tầng client.

## Lỗi / lưu ý đã biết (chưa sửa — nằm ngoài phạm vi "chỉ sửa aggregation")
- **BatchNorm + FedAvg trên non-IID cực mạnh** → thống kê BN bị hỏng (nguyên nhân nặng nhất).
- **Client drift**: local_epochs=1 nhưng client chênh số mẫu ~800 lần (4k vs 3.1M).
- Classifier là Linear thường nhưng init prototype kiểu cosine → lệch chuẩn (chưa dùng cosine classifier).
- Chống quên yếu (L2 anchor về base, không KD/replay trong Phase 2).
- Robust hơn FedAvg **không được đảm bảo** — phụ thuộc `acc_i` phân biệt client tốt hay không.

## Việc tiếp theo có thể làm
Chạy Kaggle `--debug` → full; so robust vs fedavg. Nếu muốn cải thiện thật: GroupNorm thay BN, cosine classifier, cân số bước client, KD+replay.
