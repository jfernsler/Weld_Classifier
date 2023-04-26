import os
from pathlib import Path
import torch

SCRIPT_PATH = Path(__file__).absolute()
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)

DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'ss304')

def print_device_info():
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))

def train_model(model_name='ss304_weld_model', epochs=10):
    print('train_model')

def confusion_matrix(model_name='weld_resnet50_model'):
    print('confusion_matrix')

def eval_model(model_name='weld_resnet50_model', batch_size=4):
    print('eval_model')