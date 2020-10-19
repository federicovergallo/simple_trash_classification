import requests
import pandas as pd
import tqdm
import os

# Params
csv_file = "csv_files/smashed_plastic_bottles.csv"
dataset_folder = 'plastic/'
fn = 'plastic'
# Add mode if want to add more samples to a folder
add_mode = True

df = pd.read_csv(csv_file, sep="\n")
for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
    url = row.values[0]
    if add_mode:
        index += len(os.listdir(dataset_folder))
    file_name = fn+"_"+str(index)+'.jpg'
    response = requests.get(url)
    file = open(dataset_folder+file_name, "wb")
    file.write(response.content)
    file.close()