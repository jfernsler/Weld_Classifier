import os, sys
from pathlib import Path
import torch
import argparse

from ss304_train import run_train
from ss304_stats import check_model_single, check_model_batch, check_accuracy_performance

SCRIPT_PATH = Path(__file__).absolute()
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)

DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'ss304')

def print_device_info():
    print(torch.__version__)
    print(torch.cuda.get_device_name(0))

def main(args):
    if args.eval_single:
        print('*'*10, ' Single Image Evaluation ', '*'*10)
        check_model_single()
    elif args.eval_batch:
        print('*'*10, ' Batch Image Evaluation ', '*'*10)
        check_model_batch()
    elif args.eval_timing:
        print('*'*10, ' 50 Image Timing Check ', '*'*10)
        check_accuracy_performance(size=50)
    elif args.train:
        print('*'*10, ' Training On Reduced Set ', '*'*10)
        run_train(epochs=20)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='CS614 Assignment 1 - Welding Defect Detection')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-es', '--eval_single', 
                        action='store_true', help='Evaluate a single image on the weld_resnet50_model model fine tuned for this class')
    group.add_argument('-eb', '--eval_batch', 
                        action='store_true', help='Evaluate a batch of images on the weld_resnet50_model model fine tuned for this class')
    group.add_argument('-et', '--eval_timing', 
                        action='store_true', help='Evaluate the timing and accuracy over 50 images of the weld_resnet50_model model fine tuned for this class')
    group.add_argument('-t', '--train', 
                        action='store_true', help='Will train the model on the reduced dataset for 20 epochs as v6')
    
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    main(args)
