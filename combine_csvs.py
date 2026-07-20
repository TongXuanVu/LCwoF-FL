import os
import pandas as pd

base_dir = r"C:\FederatedLearning\LCwoF-FL\logs"

scenarios = [
    ("FullData", os.path.join(base_dir, "lcwof_fl_ciciot23", "seed42_30-06-26_14-51")),
    ("1% Few-Shot", os.path.join(base_dir, "fewshot1%")),
    ("10-Shot", os.path.join(base_dir, "10shot"))
]

# Process round 30
df_30_list = []
for name, dir_path in scenarios:
    file_name = f"LCwoF_FL_{name.replace(' ', '').replace('%', 'Percent').replace('-', '')}_round30_v2.csv"
    if name == "1% Few-Shot":
        file_name = "LCwoF_FL_1Percent_round30_v2.csv"
    elif name == "10-Shot":
        file_name = "LCwoF_FL_10Shot_round30_v2.csv"
        
    path = os.path.join(dir_path, file_name)
    if os.path.exists(path):
        df = pd.read_csv(path)
        df.insert(0, "scenario", name)
        df_30_list.append(df)

if df_30_list:
    combined_30 = pd.concat(df_30_list, ignore_index=True)
    scenario_order = ["FullData", "1% Few-Shot", "10-Shot"]
    combined_30['scenario'] = pd.Categorical(combined_30['scenario'], categories=scenario_order, ordered=True)
    combined_30 = combined_30.sort_values(by=['task_id', 'scenario'])
    out_30_path = os.path.join(base_dir, "LCwoF_FL_Combined_Round30_v3.csv")
    combined_30.to_csv(out_30_path, index=False)
    print(f"Saved combined Round 30 metrics to: {out_30_path}")

# Process 180 rounds
df_180_list = []
for name, dir_path in scenarios:
    file_name = f"LCwoF_FL_{name.replace(' ', '').replace('%', 'Percent').replace('-', '')}_180_rounds_v2.csv"
    if name == "1% Few-Shot":
        file_name = "LCwoF_FL_1Percent_180_rounds_v2.csv"
    elif name == "10-Shot":
        file_name = "LCwoF_FL_10Shot_180_rounds_v2.csv"
        
    path = os.path.join(dir_path, file_name)
    if os.path.exists(path):
        df = pd.read_csv(path)
        df.insert(0, "scenario", name)
        df_180_list.append(df)

if df_180_list:
    combined_180 = pd.concat(df_180_list, ignore_index=True)
    out_180_path = os.path.join(base_dir, "LCwoF_FL_Combined_180Rounds_v3.csv")
    combined_180.to_csv(out_180_path, index=False)
    print(f"Saved combined 180 rounds metrics to: {out_180_path}")

