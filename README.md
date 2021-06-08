# CNN Sensitivity Analysis

In this repository I will be testing different combinations of neural networks, learning rate and optimizer for classifying histopathologic images (oral cancer), in order to compare accuracy, recall and precision of the different combinations. The data used for this analysis are histopathologic images of the oral cavity from the [Histopathological imaging database for oral cancer analysis](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6994517/) paper by Ayursundra Healthcare Pvt e Dr. B. Borooah Cancer Research Institute.

The images from both sets were mixed and then created patches of batch size 30. This creates a few problems due to the fact that there are some patches that have nothing (are white/gray).

Patches of normal oral cavatiy.
![Normal collage](https://github.com/beamaia/ic-neural-networks-sensitivity-analysis/blob/main/images/patches/normal-collage.png?raw=true)


Patches of carncerous oral cavatiy.
![Carcinoma collage](https://github.com/beamaia/ic-neural-networks-sensitivity-analysis/blob/main/images/patches/carcinoma-collage.png?raw=true)

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

The learning rates used in this analysis are changes acording to learning rate StepLR from PyTorch. It starts at 0.001 and decreases at each 20 steps by 0.5. The loss function is weighted.
