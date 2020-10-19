import os
import tqdm

folders = ['metal', 'plastic_web']
names = ['cans', 'plastic']
for folder, name in zip(folders, names):
    fileList = os.listdir(folder)
    for file in tqdm.tqdm(fileList):
        index = file.split("_")[1]
        path = folder+"/"+file
        new_path = folder+"/"+name+"_"+index
        os.rename(path, new_path)