import os
import pandas as pd

def process_logs_all_phases(scenario_name, round_csv_path, task1_csv_path=None):
    df_list = []
    
    # Task 1 was pre-trained and its logs are in the fulldata folder
    if task1_csv_path and os.path.exists(task1_csv_path):
        df_task1 = pd.read_csv(task1_csv_path, header=None)
        if df_task1.iloc[0, 0] == 'round_idx':
            df_task1.columns = df_task1.iloc[0]
            df_task1 = df_task1[1:]
        else:
            df_task1.columns = ["round_idx", "task", "phase", "round", "train_loss", "test_loss", 
                                "acc", "prec_mic", "prec_mac", "prec_wei", 
                                "rec_mic", "rec_mac", "rec_wei", 
                                "f1_mic", "f1_mac", "f1_wei", "fpr"]
        
        df_task1['task'] = df_task1['task'].astype(int)
        df_task1['phase'] = df_task1['phase'].astype(int)
        # Get all of Task 1 (should be Phase 1)
        t1 = df_task1[df_task1['task'] == 1]
        df_list.append(t1)

    if os.path.exists(round_csv_path):
        df_main = pd.read_csv(round_csv_path, header=None)
        if df_main.iloc[0, 0] == 'round_idx':
            df_main.columns = df_main.iloc[0]
            df_main = df_main[1:]
        else:
            df_main.columns = ["round_idx", "task", "phase", "round", "train_loss", "test_loss", 
                                "acc", "prec_mic", "prec_mac", "prec_wei", 
                                "rec_mic", "rec_mac", "rec_wei", 
                                "f1_mic", "f1_mac", "f1_wei", "fpr"]
        
        df_main['task'] = df_main['task'].astype(int)
        df_main['phase'] = df_main['phase'].astype(int)
        
        # Get all tasks > 1, ALL PHASES
        t_others = df_main[df_main['task'] > 1]
        df_list.append(t_others)
    
    if not df_list:
        print(f"No data found for {scenario_name}")
        return
        
    df_combined = pd.concat(df_list, ignore_index=True)
    
    # Format and select columns for ALL rounds (330 rounds)
    # Headers required: round_idx, task, phase, epoch, acc, prec_mic, prec_mac, prec_wei, rec_mic, rec_mac, rec_wei, f1_mic, f1_mac, f1_wei, loss
    
    out_all = pd.DataFrame()
    out_all['scenario'] = [scenario_name] * len(df_combined)
    out_all['round_idx'] = df_combined['round_idx'].astype(int)
    out_all['task'] = df_combined['task'].astype(int)
    out_all['phase'] = df_combined['phase'].astype(int)
    out_all['epoch'] = df_combined['round'].astype(int)
    
    metrics = ['acc', 'prec_mic', 'prec_mac', 'prec_wei', 'rec_mic', 'rec_mac', 'rec_wei', 'f1_mic', 'f1_mac', 'f1_wei']
    for m in metrics:
        vals = df_combined[m].astype(float)
        out_all[m] = vals.apply(lambda x: f"{x:.2f}")
        
    # Extract Train Loss
    out_all['loss'] = df_combined['train_loss'].astype(float).apply(lambda x: f"{x:.4f}")
    
    return out_all

base_dir = r"C:\FederatedLearning\LCwoF-FL\logs"
# Full data baseline logs
full_data_dir = os.path.join(base_dir, "lcwof_fl_ciciot23", "seed42_30-06-26_14-51")

scenarios = [
    ("FullData", os.path.join(full_data_dir, "round_metrics.csv"), os.path.join(full_data_dir, "round_metrics.csv")),
    ("1% Few-Shot", os.path.join(base_dir, "fewshot1%", "round_metrics.csv"), os.path.join(full_data_dir, "round_metrics.csv")),
    ("10-Shot", os.path.join(base_dir, "10shot", "round_metrics.csv"), os.path.join(full_data_dir, "round_metrics.csv"))
]

all_dfs = []
for name, p_csv, p_task1 in scenarios:
    df = process_logs_all_phases(name, p_csv, p_task1)
    if df is not None:
        all_dfs.append(df)

if all_dfs:
    final_combined = pd.concat(all_dfs, ignore_index=True)
    out_path = os.path.join(base_dir, "LCwoF_FL_Combined_All_330_Rounds.csv")
    final_combined.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
