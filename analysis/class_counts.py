import argparse
import pandas as pd 

parser = argparse.ArgumentParser(description="Count number of patients with each label")
parser.add_argument('label_file', help="Csv file with labels")
parser.add_argument('--filter', help="Txt file with patient ids")
args = parser.parse_args()

labels = pd.read_csv(args.label_file)
if args.filter:
    with open(args.filter) as f:
        patients = [int(patient.strip()) for patient in f]
    labels = labels[labels['patient_id'].isin(patients)]
print(labels.sum())
