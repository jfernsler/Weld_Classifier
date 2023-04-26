from pathlib import Path
import os, random
from PIL import Image

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
    device = get_device(show=False)
    model = ss304_weld_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def check_accuracy(model_name='weld_resnet50_model'):
    data, data_size = get_dataset(type='test', loader=True, batch_size=32)
    model = load_model(os.path.join(SCRIPT_DIR, '..', 'models', f'{model_name}.pt'))

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


def check_model(model_name='weld_resnet50_model'):

    dataset = get_dataset(type='test')

    rand_idx = random.randint(0, len(dataset))

    img_data = dataset[rand_idx]
    img = img_data['image']
    class_actual = dataset.get_class(rand_idx)

    # img.show()

    model = load_model(os.path.join(SCRIPT_DIR, '..', 'models', f'{model_name}.pt'))

    # Use the model and print the predicted category
    prediction = model(img.unsqueeze(0)).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    class_pred = dataset.classes[class_id]

    print(f'Img Index: {rand_idx}')
    print(f"Actual: {class_actual} || Prediction: {class_pred}: {100 * score:.1f}%")


def check_model_batch(model_name='weld_resnet50_model', batch_size=9):

    dataset = get_dataset(type='test')
    dataset_loader, _ = get_dataset(type='test', loader=True, batch_size=batch_size)

    rand_idx = random.randint(0, len(dataset))
    img_data = dataset[rand_idx]
    img = img_data['image']
    class_actual = dataset.get_class(rand_idx)

    class_list = dataset.classes

    # img.show()

    model = load_model(os.path.join(SCRIPT_DIR, '..', 'models', f'{model_name}.pt'))

    for bi, d in enumerate(dataset_loader):
        img = d['image']
        labels = d['label']
        class_actual = d['class']
        path = d['path']
        prediction = model(img).softmax(1)
        break

    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    # for i in range(1, cols * rows + 1):
    #     sample_idx = torch.randint(len(training_data), size=(1,)).item()
    #     img, label = training_data[sample_idx]
    #     figure.add_subplot(rows, cols, i)
    #     plt.title(labels_map[label])
    #     plt.axis("off")
    #     plt.imshow(img.squeeze(), cmap="gray")
    # plt.show()

    print('Actual\t\t\tPrediction\t\tScore')
    for i in range(len(prediction)):
        y = prediction[i]
        x = class_actual[i]
        y_pred = y.argmax().item()
        pred_class = class_list[y_pred]
        s = y[y_pred].item() * 100
        acc_class = x
        print(f'{acc_class}\t\t\t{pred_class}\t\t{s:.2f}%')

        img = Image.open(path[i])
        figure.add_subplot(rows, cols, i+1)
        plt.title(f'{acc_class} {s:.2f}%\nActual: {pred_class}')
        plt.axis("off")
        plt.imshow(img)
    plt.show()
    

def make_matrix(model_name='weld_resnet50_model'):
    BATCH_SIZE = 64
    device = get_device(show=True)

    # create the pytorch data loader
    test_dataset = get_dataset(data_type='test')
    test_dataset_loader, td_size = get_dataset(data_type='test', loader=True, batch_size=BATCH_SIZE)

    model_path = os.path.join(SCRIPT_DIR, '..', 'models', f'{model_name}.pt')
    figure_path = os.path.join(SCRIPT_DIR, '..', 'charts', f'{model_name}_confusion.png')

    model = ss304_weld_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    y_pred = []
    y_true = []

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
    plt.savefig(figure_path)

    return


if __name__ == '__main__':
    check_model_batch()
    #check_accuracy()
    #make_matrix()
    # print('Done.