# Welding image classifier



## Pitch:
Large scale additive metal manufacturing is a quickly growing field with companies such as Relativity Space and Rosotics building new machines to create very large, critical structures for various industries. Printing such structure requires high tolerances and close observation of the layers to ensure quality, flag issues, and be able to increase manufacturing speeds. I propose a CNN based vision network trained specifically to identify errors just outside of the welding bead in order to provide a realtime feedback loop for the manufacturing system and identify potential issues with a part and there locations which may require closer inspection.
## Data source:
Finding data for such a project is difficult due to the niche nature and the incredibly bright welding process. I did, however, find a data source on Kaggle which contains 10GB of labeled stainless steel welding images capture with and an HDR camera each identified with the following classifications:
1. good weld
2. burn through
3. contamination
4. lack of fusion
5. lack of shielding gas 6. high travel speed
For potential errors in an additive process ‘lack of fusion’ and ‘high speed travel’ are particularly relevant.

## Model and Data Justification:
I chose to fine-tune a ResNet50 given it’s high accuracy, light weight, and it’s fast inference time given that this would be designed for a realtime feedback system. I also chose a deeper network given the similarity in the images across the different classes with the hope that it would resolve more of those subtle differences.
     
## Commented Examples
Nine randomly selected images from the dataset, the predicted values, confidence and actuals. Generated using the ‘—eval_batch’ flag.

![Selected images and their classifications](/charts/batch_eval.png)

Here you can see the similarity of the images across some of the classes. The network needs to focus on the weld pool following the electrode in order to make the predictions which it did once I found the proper training parameters.
 
## Testing

![Confusion matrix](/charts/weld_resnet50_model_v6_confusion_v6.png)

The dataset had 11,160 images set aside out of the training and validation set for testing, which I applied the model to for the above confusion matrix. Overall the results are very good - especially in the classes I was hoping for the most success in. Additionally the very high ‘good weld’ classification helps build confidence in true positives.

![Accuracy over Epochs](/charts/Accuracy_v6.png)

Accuracy converged very quickly - though I did find that with an increase in epochs, the potential of overfitting would occur pretty soon.
  
## Code and Instructions to Run it
The code, model, and a reduced set of data can be cloned from: * https://github.com/jfernsler/weld_classifier
Once cloned the primary script to run is in /src/a1_main.py. Run this script with one of the following flags:
* a1_main.py -es
    * --eval_single
    * Evaluate a single image on the weld_resnet50_model model fine tuned for this class
* a1_main.py -eb
    * -eval_batch
    * Evaluate a batch of images on the weld_resnet50_model model fine tuned for this class
* a1_main.py -et
    * -eval_timing
    * Evaluate the timing and accuracy over 50 images of the weld_resnet50_model model fine tuned for this class
* a1_main.py -t
    * --train
    * Will train a new model on the reduced dataset for 20 epochs as v6 as a test. It won’t result in a well trained network given the small dataset, but it shows the process.
### Addendum - links from the document:
* Code:
    * https://github.com/jfernsler/weld_classifier
* Dataset - TIG Stainless Steel 304:
    * https://www.kaggle.com/datasets/danielbacioiu/tig-stainless-steel-304
* Linked Companies: 
    * Relativity Space:
        * https://www.relativityspace.com/stargate 
    * Rosotics:
        * https://www.rosotics.com/