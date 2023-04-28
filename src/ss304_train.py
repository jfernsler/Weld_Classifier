import os

import torch.nn as nn
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchmetrics.classification import BinaryAccuracy

import numpy as np

from ss304_model import ss304_weld_model
from ss304_utils import get_device, get_dataset
from ss304_stats import make_charts

from ss304_globals import *

BASE_NAME = 'weld_resnet50_model'
VERSION = 'v7' # up from v6
CSV_NAME = f'{BASE_NAME}_{VERSION}.csv'
EPOCH_CSV_PATH = os.path.join(CSV_DIR, CSV_NAME)

def train_model(model, 
                train_loader, 
                train_size,
                valid_loader,
                valid_size,
                optimizer, 
                scheduler, 
                num_epochs):
    
    device = get_device(show=False)

    epoch_train_loss = 1
    epoch_valid_loss = 1

    train_model_loss = []
    train_model_acc = []
    valid_model_loss = []
    valid_model_acc = []

    fe = open(EPOCH_CSV_PATH,'w')
    fe.write('epoch, train_loss, train_accuracy, valid_loss, valid_accuracy\n')
    fe.write('0,1,0,1,0\n')
    fe.close()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters())
    metric = BinaryAccuracy().to(device)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model.train()

        # just append all things to lists and avg at the end
        # within numpy?
        running_train_loss = []
        running_valid_loss = []
        running_train_accuracy = []
        running_valid_accuracy = []

        # Iterate over train data.
        for bi, d in enumerate(train_loader):
            inputs = d['image']
            labels = d['label']
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_train_loss.append(loss.item())
            running_train_accuracy.append(metric(outputs, labels).cpu())

            print('TRAINING:')
            print(f'Epoch: {epoch} Batch: {bi}:')
            print(f'\tbatch loss: {running_train_loss[-1]}')
            print(f'\tbatch accuracy: {running_train_accuracy[-1]}')
            print(f'\tavg loss: {np.mean(running_train_loss)}')
            print(f'\tavg accuracy: {np.mean(running_train_accuracy)}')

        model.eval()

        # Iterate over valid data.
        for bi, d in enumerate(valid_loader):
            inputs = d['image']
            labels = d['label']
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            running_valid_loss.append(loss.item())
            running_valid_accuracy.append(metric(outputs, labels).cpu())

            print('VALIDATION:')
            print(f'Epoch: {epoch} Batch: {bi}:')
            print(f'\tbatch loss: {running_valid_loss[-1]}')
            print(f'\tbatch accuracy: {running_valid_accuracy[-1]}')
            print(f'\tavg loss: {np.mean(running_valid_loss)}')
            print(f'\tavg accuracy: {np.mean(running_valid_accuracy)}')

        scheduler.step()
        
        epoch_train_loss = np.mean(running_train_loss)
        epoch_valid_loss = np.mean(running_valid_loss)
        epoch_train_accuracy = np.mean(running_train_accuracy)
        epoch_valid_accuracy = np.mean(running_valid_accuracy)

        train_model_loss.append(epoch_train_loss)
        valid_model_loss.append(epoch_valid_loss)
        train_model_acc.append(epoch_train_accuracy)
        valid_model_acc.append(epoch_valid_accuracy)

        fe = open(EPOCH_CSV_PATH,'a')
        fe.write(f'{epoch},{epoch_train_loss},{epoch_train_accuracy},{epoch_valid_loss},{epoch_valid_accuracy}\n')
        fe.close()

        print('*' * 80)
        print(f'Epoch: {epoch}: Train Loss: {epoch_train_loss}')
        print(f'Epoch: {epoch}: Valid Loss: {epoch_valid_loss}')
        print(f'Epoch: {epoch}: Train Accuracy: {epoch_train_accuracy}')
        print(f'Epoch: {epoch}: Valid Accuracy: {epoch_valid_accuracy}')
        print('*' * 80)

    make_charts(EPOCH_CSV_PATH, VERSION)
    print('*' * 80 )
    print('Training complete')
    print('*' * 80 )

    return model


def run_train(epochs=1):
    device = get_device(show=False)
    print(f'Device : {device}')
    
    # get the datasets
    train_dataset_loader, train_size = get_dataset(type='train', loader=True, batch_size=32)
    valid_dataset_loader, valid_size = get_dataset(type='valid', loader=True, batch_size=32)

    # get the model
    model_weld = ss304_weld_model()
    # push model to device
    model_weld = model_weld.to(get_device())

    # set up optimizer and learning rate scheduler
    optimizer_sat = optim.Adam(model_weld.fc.parameters(), lr=0.1)
    lr_sch = lr_scheduler.StepLR(optimizer_sat, step_size=2, gamma=0.1)

    # begin training
    model_weld = train_model(model_weld,
                        train_dataset_loader,
                        train_size,
                        valid_dataset_loader,
                        valid_size,
                        optimizer_sat,
                        lr_sch,
                        num_epochs=epochs)

    # define a path and save it out
    print(f'Saving model {BASE_NAME}_{VERSION}.pt to {MODEL_DIR}')
    model_path = os.path.join(MODEL_DIR, f'{BASE_NAME}_{VERSION}.pt')
    torch.save(model_weld.state_dict(), model_path)
    print('Model saved, Training complete')


def list_model(model):
    model_dict = model.state_dict()
    dict_keys = list(model_dict.keys())
    for key in dict_keys:
        print(f'{key} : {model_dict[key].shape}')


if __name__ == '__main__':
    run_train(epochs=10)

