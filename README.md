# CNN Sensitivity Analysis

In this repository I will be testing different combinations of neural networks, learning rate and optimizer for classifying histopathologic images (oral cancer), in order to compare accuracy, recall and precision of the different combinations. The data used for this analysis are histopathologic images of the oral cavity from the [Histopathological imaging database for oral cancer analysis](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6994517/) paper by Ayursundra Healthcare Pvt e Dr. B. Borooah Cancer Research Institute.

This analysis was made using Python 3.8.5 and PyTorch 1.7.1.

The models used in this sensitivity analysis are:
- ResNet50
- MobileNetV2
- InceptionV3
- VGG16
- DenseNet121

The optimizers used in this sensitivity analysis are:
- Adam 
- SGD

The learning rates used in this analysis are:
- 0.001
- 0.0001
- 0.00001

The loss function is weighted.
