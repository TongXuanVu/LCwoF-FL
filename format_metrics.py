import os
import pandas as pd

def process_logs(scenario_name, round_csv_path, task1_csv_path=None):
    df_list = []
    
    # Task 1 was pre-trained and its logs are in the fulldata folder
    if task1_csv_path and os.path.exists(task1_csv_path):
        df_task1 = pd.read_csv(task1_csv_path, header=None)
        # Add column names if they are missing
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
        # Filter Task 1 Phase 1
        t1 = df_task1[(df_task1['task'] == 1) & (df_task1['phase'] == 1)]
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
        
        # Filter Task 2-6 Phase 3
        t_others = df_main[(df_main['task'] > 1) & (df_main['phase'] == 3)]
        df_list.append(t_others)
    
    if not df_list:
        print(f"No data found for {scenario_name}")
        return
        
    df_combined = pd.concat(df_list, ignore_index=True)
    
    # Format and select columns for 180 rounds
    # Headers required: task_id, round_in_task, global_round, acc, prec_mic, prec_mac, prec_wei, rec_mic, rec_mac, rec_wei, f1_mic, f1_mac, f1_wei, loss
    
    out_180 = pd.DataFrame()
    out_180['task_id'] = df_combined['task'].astype(int)
    out_180['round_in_task'] = df_combined['round'].astype(int)
    out_180['global_round'] = [(t - 1) * 30 + r for t, r in zip(out_180['task_id'], out_180['round_in_task'])]
    
    metrics = ['acc', 'prec_mic', 'prec_mac', 'prec_wei', 'rec_mic', 'rec_mac', 'rec_wei', 'f1_mic', 'f1_mac', 'f1_wei']
    for m in metrics:
        # Convert to float and format to 2 decimal places. 
        # Check if already percentage (e.g. > 1.0)
        vals = df_combined[m].astype(float)
        # Values in LCwoF logs are already in percentage (0-100)
        out_180[m] = vals.apply(lambda x: f"{x:.2f}")
        
    out_180['loss'] = df_combined['train_loss'].astype(float).apply(lambda x: f"{x:.4f}")
    
    out_180_path = os.path.join(os.path.dirname(round_csv_path), f"LCwoF_FL_{scenario_name}_180_rounds_v2.csv")
    out_180.to_csv(out_180_path, index=False)
    print(f"Saved {out_180_path}")
    
    # Format and select columns for Round 30 only
    # Headers required: accuracy, f1_micro, f1_macro, f1_weight, precision_micro, precision_macro, precision_weight, recall_micro, recall_macro, recall_weight, loss
    
    df_30 = out_180[out_180['round_in_task'] == 30].copy()
    
    out_30 = pd.DataFrame()
    out_30['task_id'] = df_30['task_id']
    out_30['accuracy'] = df_30['acc']
    out_30['f1_micro'] = df_30['f1_mic']
    out_30['f1_macro'] = df_30['f1_mac']
    out_30['f1_weight'] = df_30['f1_wei']
    out_30['precision_micro'] = df_30['prec_mic']
    out_30['precision_macro'] = df_30['prec_mac']
    out_30['precision_weight'] = df_30['prec_wei']
    out_30['recall_micro'] = df_30['rec_mic']
    out_30['recall_macro'] = df_30['rec_mac']
    out_30['recall_weight'] = df_30['rec_wei']
    out_30['loss'] = df_30['loss']
    
    out_30_path = os.path.join(os.path.dirname(round_csv_path), f"LCwoF_FL_{scenario_name}_round30_v2.csv")
    out_30.to_csv(out_30_path, index=False)
    print(f"Saved {out_30_path}")

base_dir = r"C:\FederatedLearning\LCwoF-FL\logs"
# Full data baseline logs
full_data_dir = os.path.join(base_dir, "lcwof_fl_ciciot23", "seed42_30-06-26_14-51")

scenarios = [
    ("FullData", os.path.join(full_data_dir, "round_metrics.csv"), os.path.join(full_data_dir, "round_metrics.csv")),
    ("1Percent", os.path.join(base_dir, "fewshot1%", "round_metrics.csv"), os.path.join(full_data_dir, "round_metrics.csv")),
    ("10Shot", os.path.join(base_dir, "10shot", "round_metrics.csv"), os.path.join(full_data_dir, "round_metrics.csv"))
]

for name, p_csv, p_task1 in scenarios:
    process_logs(name, p_csv, p_task1)
