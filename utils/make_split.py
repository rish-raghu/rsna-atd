import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description="Generate train/val split")
parser.add_argument('val_frac', type=float)
parser.add_argument('split_name')
parser.add_argument('--injured', action='store_true')
parser.add_argument('--seed', type=int, default=None)
args = parser.parse_args()

df = pd.read_csv("data/train.csv")
if args.injured:
    df = df[df["any_injury"]==1]
patients = np.array(df["patient_id"])
if args.seed: np.random.seed(args.seed)
np.random.shuffle(patients)
num_val = int(len(patients)*args.val_frac)
val_patients = patients[:num_val]
train_patients = patients[num_val:]

with open(f"data/{args.split_name}.val.txt", "w") as f:
    for patient in val_patients:
        f.write(f"{patient}\n")
f.close()

with open(f"data/{args.split_name}.train.txt", "w") as f:
    for patient in train_patients:
        f.write(f"{patient}\n")
f.close()
