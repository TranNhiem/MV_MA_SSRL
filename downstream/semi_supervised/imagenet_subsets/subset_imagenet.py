from genericpath import exists
import os, shutil
from pydoc import classname

SIZE = '10per'

imagenet_path = '/data1/1K_New/train'

if SIZE == '1per':
    images_path = '/downstream/semi_supervised/imagenet_subsets/1percent.txt'
    write_path = '/data1/1K_New/train_1per'
elif SIZE == '10per':
    images_path = '/downstream/semi_supervised/imagenet_subsets/10percent.txt'
    write_path = '/data1/1K_New/train_10per'

class_names = [x for x in os.listdir(imagenet_path) if '.tar' not in x]

for class_name in class_names:
    if not os.path.exists(os.path.join(write_path, class_name)):
        os.mkdir(os.path.join(write_path, class_name))

with open(images_path, 'r') as f:
    images_names = f.readlines()

images_names[len(images_names) - 1] = images_names[len(images_names) - 1] + '\n'

for name in images_names:
    name = name[:-1]
    label = name[:9]
    image_path = os.path.join(imagenet_path, label, name)
    dest_path = os.path.join(write_path, label, name)
    if not os.path.exists(dest_path):
        shutil.copyfile(image_path, dest_path)


