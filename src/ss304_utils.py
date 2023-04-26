import torch
from torchvision import transforms

from ss304_dataset import ss304Dataset

def get_device(show=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if show:
        print('#'*80)
        print(f'Using device: {device}')
        print('#'*80)
    return device

def get_dataset(type='test', loader=False, batch_size=32):
    IMAGE_SIZE = 224
    BATCH_SIZE = batch_size

    # mean and std of imagenet dataset
    IMG_MEAN = torch.tensor([0.485, 0.456, 0.406])
    IMG_STD = torch.tensor([0.229, 0.224, 0.225])

    # make some augmentations on training data
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE)),
        #transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD)
    ])

    # use the collections dataset class we created earlier
    dataset = ss304Dataset(data_type=type,
                                    transform=train_transform)
    
    if not loader:
        return dataset

    # create the pytorch data loader
    dataset_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=BATCH_SIZE,
                                                shuffle=True,
                                                num_workers=8)
    
    return dataset_loader, len(dataset)