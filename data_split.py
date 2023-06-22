import glob
import os
import random
import shutil

files = glob.glob1("data/train_data","*.txt")
files_counter = len(files)
val_count = int(files_counter*0.2)
selected = random.sample(files, val_count)

for file in selected:
    file = file.split('.txt')[0]
    shutil.move("data/train_data/" + file + ".txt", "data/val_data/" + file + ".txt")
    if os.path.exists("data/train_data/" + file + ".jpg"):
        shutil.move("data/train_data/" + file + ".jpg", "data/val_data/" + file + ".jpg")

    else:
        shutil.move("data/train_data/" + file + ".png", "data/val_data/" + file + ".png")
