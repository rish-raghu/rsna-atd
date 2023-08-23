import pandas as pd
import os

rows = []
for patient in os.listdir("data/train_pngs"):
    for scan in os.listdir(os.path.join("data/train_pngs", patient)):
        count = len(os.listdir(os.path.join("data/train_pngs", patient, scan)))
        rows.append((patient, scan, count))
df = pd.DataFrame(rows, columns=['patient_id', 'scan_id', 'image_count'])
#df.to_csv("data/train_image_counts.csv")
print("Min count: ", min(df['image_count']))
print("Max count: ", max(df['image_count']))
print("Mean count: ", df['image_count'].mean())
