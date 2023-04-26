# Weld Quality Image Classifier

This is a fine tuned model based on the resnet50 base model and trained on the ss304 dataset from kaggle.

Provided is the fine-tuned model, a small sample of data, and all of the required scripts.

requires torch, torchvision, pandas, matplotlib

to run execute *a1_main.py* with one of the following flags:
* -es or --eval_single
    * Evaluate a single image on the weld_resnet50_model model fine tuned for this class
* -eb or --eval_batch
    * Evaluate a batch of images on the weld_resnet50_model model fine tuned for this class
* -et or --eval_timing
    * Evaluate the timing and accuracy over 50 images of the weld_resnet50_model model fine tuned for this class
* -t or --train
    * Will train the model on the reduced dataset for 20 epochs as v6 as a test

The code is modified within the dataset python script to use the reduced data set. If you download the full dataset the data path will need to be updated in that file.