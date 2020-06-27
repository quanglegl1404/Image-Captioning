import json
import os
from shutil import copyfile


file = open("D:\\Learning\\Thesis\\Origin\\Quang-Image-Captions\\data\\annotations\\uitviic_captions_val2017.json", 'r',encoding="utf8")
# print(file.readlines())
json_data = json.load(file)
source = "D:\\Learning\\Thesis\\Origin\\Image-Captions\\data\\train2017"
dest = "D:\\Learning\\Thesis\\Origin\\Image-Captions\\data\\sportball_val_2017"

for x in json_data['images']:

    path = os.path.join(source, x['file_name'])
    dst = os.path.join(dest, x['file_name'])
    #print(dst)
    copyfile(path, dst)