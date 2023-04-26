from PIL import Image
from torch.utils.data import Dataset
import torch
import os, json

from ss304_globals import *

#DATA_PATH = os.path.join(DATA_DIR, 'ss304')
DATA_PATH = os.path.join(DATA_DIR, 'ss304_reduced')

CLASS_LIST = ['good weld', 'burn through', 'contamination', 'lack of fusion', 'lack of shielding gas', 'high travel speed']

class ss304Dataset(Dataset):
    """
    Main dataset class for ss304 weld images.
    """
    def __init__(self, 
                 root_dir = DATA_PATH,
                 data_type = 'test', 
                 transform=None):
        
        self.classes = CLASS_LIST
        self.transform = transform

        self.root_dir = os.path.join(root_dir, data_type)
        json_path = os.path.join(self.root_dir, data_type + '.json')

        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.data = dict()

        for idx, image in enumerate(data.keys()):
            self.data[idx] = {'image': os.path.join(self.root_dir, image), 
                              'label': data[image],
                              'class': CLASS_LIST[data[image]],}

        print(f'Loaded {len(self.data)} images from {data_type} dataset.')

        self.num_classes = len(CLASS_LIST)

                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data[idx]['image']

        # convert grayscale to rgb
        image = Image.open(img_name).convert('RGB')

        label = self.data[idx]['label']

        label_tensor = torch.zeros(self.num_classes)
        label_tensor[label] = 1

        if self.transform:
            image = self.transform(image)

        return {'image': image,
                'label': label_tensor,
                'class': self.data[idx]['class'],
                'path': img_name
                }

    def check_image(self, idx):
        img_name = self.data[idx]['image']
        print(self.data[idx])
        image = Image.open(img_name)
        image.show()

    def get_img_info(self, idx):
        return self.data[idx]

    def get_class_count(self):
        return self.num_classes
    
    def get_class(self, idx):
        return self.data[idx]['class']

 