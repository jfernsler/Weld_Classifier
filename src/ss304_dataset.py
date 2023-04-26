from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import torch
import os, json

SCRIPT_PATH = Path(__file__).absolute()
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)

BASE_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'ss304')

CLASS_LIST = ['good weld', 'burn through', 'contamination', 'lack of fusion', 'lack of shielding gas', 'high travel speed']

class ss304Dataset(Dataset):
    def __init__(self, 
                 root_dir = BASE_PATH,
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

        #print('Loaded {0} images'.format(len(self.data)))

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
                'label': label_tensor
                }

    def check_image(self, idx):
        img_name = self.data[idx]['image']
        print(self.data[idx])
        image = Image.open(img_name)
        image.show()

    def get_class_count(self):
        return self.num_classes
    
    def get_class(self, idx):
        return self.data[idx]['class']

 