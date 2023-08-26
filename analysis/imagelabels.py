import pandas as pd
import os

df = pd.read_csv("data/image_level_labels.csv")

print("Number of patients: ", len(df['patient_id'].unique()))
print("Number of series: ", len(df['series_id'].unique()))
numInjuredSlices = len(df.groupby(['patient_id', 'series_id', 'instance_number']).size())
print("Number of slices with injury: ", numInjuredSlices)

count = 0
countsBySeries = df.groupby(['patient_id', 'series_id']).size().reset_index()
for patient, series in list(zip(countsBySeries['patient_id'], countsBySeries['series_id'])):
        count += len(os.listdir(f"data/train_pngs/{patient}/{series}"))
print("Total number of slices for these series: ", count)

count = 0
for patient in df['patient_id'].unique():
    for series in os.listdir(f"data/train_pngs/{patient}"):
        count += len(os.listdir(f"data/train_pngs/{patient}/{series}"))
print("Total number of slices for these patients: ", count)

bowelSlices = set(list(df[df['injury_name']=='Bowel'][['patient_id', 'series_id', 'instance_number']].itertuples(index=False, name=None)))
extravSlices = set(list(df[df['injury_name']=='Active_Extravasation'][['patient_id', 'series_id', 'instance_number']].itertuples(index=False, name=None)))
print("Number of bowel injury only slices: ", len(bowelSlices.difference(extravSlices)))
print("Number of extravasation only slices: ", len(extravSlices.difference(bowelSlices)))
print("Number of both injury slices: ", len(extravSlices.intersection(bowelSlices)))
