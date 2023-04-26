from pathlib import Path
import os, random

import torch
from torchvision import transforms
from torchmetrics.classification import BinaryAccuracy

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ss304_dataset import ss304Dataset
from ss304_model import ss304_weld_model
from ss304_utils import get_device, get_dataset

# Globals
SCRIPT_PATH = Path(__file__).absolute()
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)

def load_model(model_path):
    device = get_device()
    model = ss304_weld_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def check_accuracy():
    data, data_size = get_dataset(type='test', loader=True, batch_size=32)
    model = load_model(os.path.join(SCRIPT_DIR, '../models/weld_resnet50_model_2.pt'))

    for bi, d in enumerate(data):
        print(f'Batch {bi}')
        inputs = d['image']
        labels = d['label']
        outputs = model(inputs)

        metric = BinaryAccuracy()
        acc = metric(outputs, labels)
        print(f'Accuracy: {acc*100}%')
        if bi > 0:
            break


def check_model():
    # mean and std of imagenet dataset
    IMG_MEAN = torch.tensor([0.485, 0.456, 0.406])
    IMG_STD = torch.tensor([0.229, 0.224, 0.225])
    IMAGE_SIZE = 224

    dataset = ss304Dataset(data_type='test')

    rand_idx = random.randint(0, len(dataset))

    img_data = dataset[rand_idx]
    img = img_data['image']
    class_actual = dataset.get_class(rand_idx)

    # img.show()

    # Step 1: Initialize model with the best available weights
    # device = get_device()
    # model_path = os.path.join(SCRIPT_DIR, '../models/weld_resnet50_model_2.pt')

    # model = ss304_weld_model()
    # model.load_state_dict(torch.load(model_path, map_location=device))
    # model.to(device)
    # model.eval()
    model = load_model(os.path.join(SCRIPT_DIR, '../models/weld_resnet50_model_2.pt'))

    # Setup and apply inference preprocessing transforms
    preprocess = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD)
    ])
    batch = preprocess(img).unsqueeze(0)

    # Use the model and print the predicted category
    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    class_pred = dataset.classes[class_id]

    print(f'Img Index: {rand_idx}')
    print(f"Actual: {class_actual} || Prediction: {class_pred}: {100 * score:.1f}%")


def make_matrix():

    y_pred = []
    y_true = []

    # mean and std of imagenet dataset
    IMG_MEAN = torch.tensor([0.485, 0.456, 0.406])
    IMG_STD = torch.tensor([0.229, 0.224, 0.225])
    IMAGE_SIZE = 224
    BATCH_SIZE = 64
    
    # use the collections dataset class we created earlier
    preprocess = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD)
    ])
    test_dataset = ss304Dataset(data_type='test', transform=preprocess)

    # create the pytorch data loader
    test_dataset_loader = torch.utils.data.DataLoader(test_dataset,
                                                    shuffle=True,
                                                    batch_size=BATCH_SIZE,
                                                    num_workers=0)

    device = get_device()
    model_path = os.path.join(SCRIPT_DIR, '../models/weld_resnet50_model_5.pt')

    model = ss304_weld_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    for bi, d in enumerate(test_dataset_loader):
        # get predictions from model
        output = model(d['image'].to(device))
        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)

        # get labels
        labels = d['label']
        labels = (torch.max(torch.exp(torch.tensor(labels)), 1)[1]).data.cpu().numpy()
        y_true.extend(labels)

        print(bi, output-labels)
        # if bi > 10:
        #     break

    classes = test_dataset.classes

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None],
                            index = [classes],
                            columns = [classes])
    plt.figure(figsize = (16,12))
    plt.subplots_adjust(bottom=0.25)
    plt.title('Confusion Matrix for CNN based Weld Classification')
    hm = sn.heatmap(df_cm, annot=True, linewidths=.5, cmap='plasma', fmt='.2f', linecolor='grey')
    hm.set(xlabel='Predicted', ylabel='Truth')
    plt.savefig('confusion_matrix_5.png')

    return


if __name__ == '__main__':
    #check_model()
    #check_accuracy()
    make_matrix()
    # print('Done.