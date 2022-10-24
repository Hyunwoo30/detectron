from email.mime import image
from sklearn.model_selection import train_test_split
from glob import glob
import shutil
import os

image_files = glob("./data/*.PNG")

images = [name.replace(".png","") for name in image_files]

train_names, test_names = train_test_split(images, test_size=0.3, random_state=777, shuffle=True)
val_names, test_names = train_test_split(test_names, test_size=0.3, random_state=777, shuffle=True)

def batch_move_files(file_list, source_path, destination_path):
    for file in file_list:
        image = file.split('/')[-1] + '.PNG'
        txt = file.split('/')[-1] + '.json'
        shutil.copy(os.path.join(source_path, image), destination_path)
        shutil.copy(os.path.join(source_path, txt), destination_path)
        
    return

# source_dir = "./data/"
source_dir = "./"



test_dir = "./data/test/"
train_dir = "./data/train/"
val_dir = "./data/val/"
os.makedirs(test_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
batch_move_files(train_names, source_dir, train_dir)
batch_move_files(test_names, source_dir, test_dir)
batch_move_files(val_names, source_dir, val_dir)
