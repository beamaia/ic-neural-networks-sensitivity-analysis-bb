
# importing libraries
import torch
import torchvision
from torchvision import models
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import utils
import model 

from sklearn import metrics

def train (model_class, train_loader, val_loader, weights):
    model = model_class.model
    num_epochs = model_class.epochs
    # lr = model_class.lr
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print("Starting training...")
    print(f"Device: {device}",end="\n\n")

    criterion = model_class.loss
    optimizer = model_class.op
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5, verbose=True)

    # Creating lists
    train_accuracies = []
    train_losses = []
    y_predict = []
    val_accuracies = []
    val_losses = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1} / {num_epochs}")

        # Train epoch   
        running_loss, accuracy, y_predict_loader = train_epoch(model, optimizer, criterion, scheduler, train_loader, device)
 
        # Saves information
        # train_losses.append(running_loss) 
        # train_accuracies.append(accuracy)
        y_predict.append(y_predict_loader)

        print(f"Accuracy: {accuracy:.2f} %")
        print(f"Loss: {np.mean(train_losses):.2f}")

        # Validation 
        val_losses, val_accuracies = validation(model, criterion, val_loader, device, val_losses, val_accuracies)

    print("Finished training")

    return train_accuracies, train_losses, val_accuracies, val_losses, y_predict

def train_epoch(model, optimizer, criterion, scheduler, train_loader, device):
    torch.cuda.empty_cache()

    # Model in training mode
    model.train()
    running_loss = 0
    total_train = 0
    accuracies_train = 0
    y_predict_loader = []
    
    for _, data in enumerate(train_loader):
        images, labels = data[0].to(device), data[1].to(device)
        # Zero the parameter gradients
        optimizer.zero_grad()
        #Forward
        outputs = model(images)

        loss = criterion(outputs, labels).to(device) 
        loss.backward()

        #Optimize
        optimizer.step()

        running_loss += loss.item()

        #Train accuracy
        _, predicted = torch.max(outputs, 1)
        # y_predict_loader.append(predicted)
        total_train += labels.size(0)
        accuracies_train += (predicted == labels).sum().item()

    accuracy = accuracies_train / total_train * 100
    running_loss = running_loss/total_train
    scheduler.step()
    
    return running_loss, accuracy, y_predict_loader

def validation(model, criterion, val_loader, device, val_losses, val_accuracies):

    # Validation 
    val_loss = 0
    total_val = 0
    
    with torch.no_grad():
        accuracies_val = 0  
        running_val_loss = 0
        for _, data in enumerate(val_loader):
            images, labels = data[0], data[1]
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            val_loss = criterion(outputs, labels)

            running_val_loss += val_loss.item()
            total_val += labels.size(0)
            _, predicted = torch.max(outputs, 1)
            accuracies_val += (predicted == labels).sum().item()
    
    accuracy_val = accuracies_val / total_val * 100
    running_val = running_val_loss/total_val

    print(f"Val Accuracy: {accuracy_val:.2f} %")
    print(f"Val Loss: {running_val:.2f}", end="\n\n")
    
    # Saving validation and 
    val_losses.append(running_val)
    val_accuracies.append(accuracy_val)

    return val_losses, val_accuracies

def test(model_class, test_loader):
    print("Starting test...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_class.model
    model.to(device)
    model.eval()

    y_predict = []
    y_true = []
    accuracy = 0
    with torch.no_grad():
        total = 0
        for _, data in enumerate(test_loader):
            images, labels = data[0].to(device), data[1].to(device)
            y_true.append(labels)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_predict.append(predicted)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

    accuracy_test = accuracy / total * 100
    print(f"Accuracy: {accuracy_test}")
    print("Finished test....")
    return accuracy_test, y_predict, y_true