from pathlib import Path
import os, random
from PIL import Image
import time

import torch
from torchmetrics.classification import BinaryAccuracy

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ss304_model import ss304_weld_model
from ss304_utils import get_device, get_dataset

from ss304_globals import *


def load_model(model_path):
    device = get_device(show=False)
    model = ss304_weld_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def check_accuracy_performance(model_name='weld_resnet50_model', size=50):
    device = get_device(show=True)
    print('Checking Accuracy and Speed for SS304 Weld Model')
    print('Getting dataset...')
    tic = time.perf_counter()
    data = get_dataset(type='test')
    toc = time.perf_counter()
    print(f'Time to get dataset: {toc-tic:.4f} seconds')
    print('Loading model...')
    tic = time.perf_counter()
    model = load_model(os.path.join(MODEL_DIR, f'{model_name}.pt'))
    toc = time.perf_counter()
    print(f'Time to load model: {toc-tic:.4f} seconds')
    print('Checking accuracy and timing...')

    rand_array = np.random.randint(0, len(data), size=size)

    time_array = []
    conf_array = []
    accuracy = 0
    metric = BinaryAccuracy().to(device)
    for n, idx in enumerate(rand_array):
        input = data[idx]['image'].to(device)
        label = data[idx]['label'].to(device)
        
        tic = time.perf_counter()
        outputs = model(input.unsqueeze(0)).squeeze(0)
        toc = time.perf_counter()
        time_array.append(toc-tic)
        pred = outputs.argmax().item()
        actual = label.argmax().item()
        if pred == actual:
            accuracy += 1/float(size)
        inference = 'Correct  ' if pred == actual else 'Incorrect'
        conf_array.append(metric(outputs, label).cpu() * 100)

        print(f'{n}-> {inference} : Confidence: {conf_array[-1]:.2f}%, Time: {time_array[-1]:.4f} seconds')

    print('*'*20)
    print(f'{size} Random Samples, {accuracy*100:.2f}% Accuracy')
    print(f'Average Confidence: {np.mean(conf_array):.2f}%, Average Pred Time: {np.mean(time_array):.4f} sec')
    print('*'*20)


def check_model_single(model_name='weld_resnet50_model'):
    device = get_device(show=False)
    dataset = get_dataset(type='test')

    rand_idx = random.randint(0, len(dataset))

    img_data = dataset[rand_idx]
    img = img_data['image']
    class_actual = dataset.get_class(rand_idx)

    img_name = dataset.data[rand_idx]['image']
    Image.open(img_name).show()

    print('loading model...')
    model = load_model(os.path.join(MODEL_DIR, f'{model_name}.pt'))

    # Use the model and print the predicted category
    prediction = model(img.unsqueeze(0).to(device)).squeeze(0).softmax(0).to(device)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    class_pred = dataset.classes[class_id]

    print(f'Img Index: {rand_idx}')
    print(f"Actual: {class_actual} || Prediction: {class_pred}: {100 * score:.1f}%")


def check_model_batch(model_name='weld_resnet50_model', batch_size=9):
    device = get_device(show=False)

    dataset = get_dataset(type='test')
    dataset_loader, _ = get_dataset(type='test', loader=True, batch_size=batch_size)

    rand_idx = random.randint(0, len(dataset))
    img_data = dataset[rand_idx]
    img = img_data['image']
    class_actual = dataset.get_class(rand_idx)

    class_list = dataset.classes

    model = load_model(os.path.join(MODEL_DIR, f'{model_name}.pt'))

    for bi, d in enumerate(dataset_loader):
        img = d['image']
        labels = d['label']
        class_actual = d['class']
        path = d['path']
        prediction = model(img.to(device)).softmax(1)
        break

    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3

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
    

def make_matrix(model_name='weld_resnet50_model', version='v5'):
    BATCH_SIZE = 64
    device = get_device(show=True)

    # create the pytorch data loader
    test_dataset = get_dataset(data_type='test')
    test_dataset_loader, td_size = get_dataset(data_type='test', loader=True, batch_size=BATCH_SIZE)

    model_path = os.path.join(MODEL_DIR, f'{model_name}.pt')
    figure_path = os.path.join(CHART_DIR, f'{model_name}_confusion_{version}.png')

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


def make_epoch_chart(data, title, ylabel, figure_name, show=False):
    plt.figure(figsize=(6, 4))

    for d in data:
        plt.plot(data[d], label=d)

    plt.ylabel(ylabel)
    plt.xlabel('Epoch')
    plt.title(title)
    plt.legend()
    plt.subplots_adjust(left=0.15)
    plt.savefig(os.path.join(CHART_DIR, f'{figure_name}.png'), dpi=300, bbox_inches="tight")
    if show:
        plt.show()


def make_charts(csv_name='weld_training.csv', version='v5'):
    df = pd.read_csv(os.path.join(CSV_DIR, csv_name))
    loss = df[[' train_loss', ' valid_loss']]
    accuracy = df[[' train_accuracy', ' valid_accuracy']]
    make_epoch_chart(loss, 'Loss per Epoch', 'Loss', f'Loss_{version}', show=True)
    make_epoch_chart(accuracy, 'Accuracy per Epoch', 'Accuracy', f'Accuracy_{version}', show=False)
    

if __name__ == '__main__':
    #make_charts()
    #check_model_batch()
    check_accuracy_performance()
    #make_matrix()
