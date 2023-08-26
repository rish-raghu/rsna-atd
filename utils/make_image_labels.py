import pandas as pd
import os

imageLabels = pd.read_csv('data/image_level_labels.csv')
newRows = []
for patient in imageLabels['patient_id'].unique():
    for series in os.listdir(f'data/train_pngs/{patient}'):
        allInstances = [int(file.split('.png')[0]) for file in os.listdir(f'data/train_pngs/{patient}/{series}')]
        injuredInstances = set(list(imageLabels[(imageLabels['patient_id']==int(patient)) & (imageLabels['series_id']==int(series))]['instance_number']))
        for instance in allInstances:
            if instance not in injuredInstances:
                newRows.append((patient, series, instance, 'Healthy'))

newRows = pd.DataFrame(newRows, columns=imageLabels.columns)
imageLabels = pd.concat([imageLabels, newRows], ignore_index=True)
imageLabels = imageLabels.sort_values(["patient_id", "series_id", "instance_number"])

imageLabels['bowel'] = (imageLabels['injury_name']=='Bowel').astype(int)
imageLabels['extravasation'] = (imageLabels['injury_name']=='Active_Extravasation').astype(int)
#imageLabels['healthy'] = (imageLabels['injury_name']=='Healthy').astype(int)
imageLabels = imageLabels.drop(columns='injury_name')
imageLabels = imageLabels.drop_duplicates(subset=['patient_id', 'series_id', 'instance_number'])
imageLabels = imageLabels.sort_values(['patient_id', 'series_id', 'instance_number'])

imageLabels.to_csv('data/image_level_labels_fullinjured.csv', index=False)
