from torch.nn.modules import dropout
from torchvision import models
import torch.optim as optim
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class Model:
    def __init__(self, epochs, lr, op, model_name, version): 
        self.epochs = epochs
        self.lr = lr
        self.op_name = op
        self.model_name = model_name
        self.version = version
        self.hw = 224
        self.configure_model()
        self.configure_op()

    def configure_model(self):       
        num_classes = 2
        dropout = 0.1

        if self.model_name == "resnet50":
            self.model = models.resnet50(pretrained=True)

            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_ftrs, num_classes)
            )
            
        elif self.model_name == "mobilenetv2":
            self.model = models.mobilenet_v2(pretrained=True)

            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[0] = nn.Dropout(dropout)
            self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)

        elif self.model_name == "vgg16":
            self.model = models.vgg16(pretrained=True)

            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[5] = nn.Dropout(dropout)
            self.model.classifier[6] = nn.Linear(num_ftrs,num_classes)

        elif self.model_name == "densenet121":
            self.model = models.densenet121(pretrained=True)

            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_ftrs, num_classes)
            )
        elif self.model_name == "efficientnetb4":
            self.model =  EfficientNet.from_pretrained('efficientnet-b4')

            num_ftrs = self.model._fc.in_features
            self.model_fc = nn.Linear(num_ftrs, num_classes)
            self.model._dropout = nn.Dropout(dropout)

    def configure_loss_func(self, weight_tensor):
        self.loss = nn.CrossEntropyLoss(weight=weight_tensor)

    def configure_op(self):
        if self.op_name == "adam":
            self.op = optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.op_name == "sgd":
            self.op = optim.SGD(self.model.parameters(), lr=self.lr)
