from torchvision import models
import torch.optim as optim
import torch.nn as nn

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

        if self.model_name == "resnet50":
            self.model = models.resnet50(pretrained=True)
            self.model.fc = nn.Linear(2048, num_classes)
        elif self.model_name == "mobilenetv2":
            self.model = models.mobilenet_v2(pretrained=True)
            self.model.classifier[1] = nn.Linear(1280, num_classes)
        elif self.model_name == "inceptionv3":
            self.hw = 299
            self.model = models.inception_v3(pretrained=True, aux_logits = False)
            self.model.AuxLogits.fc = nn.Linear(768, num_classes)
            self.model.fc = nn.Linear(2048, num_classes)
        elif self.model_name == "vgg16":
            self.model = models.vgg16(pretrained=True)
            self.model.classifier[6] = nn.Linear(4096,num_classes)
        elif self.model_name == "densenet121":
            self.model = models.densenet121(pretrained=True)
            self.model.classifier = nn.Linear(1024, num_classes)
        elif self.model_name == "efficientnet":
            pass

    def configure_loss_func(self, weight_tensor):
        self.loss = nn.CrossEntropyLoss(weight=weight_tensor)

    def configure_op(self):
        if self.op_name == "adam":
            self.op = optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.op_name == "sgd":
            self.op = optim.SGD(self.model.parameters(), lr=self.lr)
