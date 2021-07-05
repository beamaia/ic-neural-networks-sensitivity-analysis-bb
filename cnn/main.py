import torch
from torchvision import models
import torch.nn as nn

import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from sklearn import metrics

from os import system
import sys
from datetime import date

import data as dt
import train as tr
import utils
import model as md

def sum_unique (x):
    unique = np.unique(x)
    sums = 0
    for _, val in enumerate(unique):
        sums += val    
    return sums, unique

def get_arg(args):
    epochs = 200
    lr = 0.0001
    model_name = "resnet50"
    version = 1
    op = "adam"

    for arg in args:
        if arg == "main.py":
            continue
        
        param, value = arg.split("=")
        if param == "epoch" or param == "epochs" or param == "num_epochs":
            epochs = int(value)
        elif param == "lr" or param == "learning_rate":
            lr = float(value)
        elif param == "model" or param == "model_name":
            model_name = value
        elif param == "version":
            version = float(value)
        elif param == "op":
            op = value

    today = date.today()
    date_today = today.strftime("%d-%m-%y")

    print("Configuration:")
    print("Epoch =", epochs, " Learning rate =", lr, " Model =", model_name, " Optimizer =", op, " Version =", version, " Date =", date_today)
    return epochs, lr, model_name, version, op, date_today

def main(date_today, epochs, lr, op, model_name, version):
    torch.cuda.empty_cache()
    model = md.Model(epochs, lr, op, model_name, version)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # images    
    (x_train, y_train), (x_val, y_val), (x_test, y_test), (normal_len, carcinoma_len) = dt.create_train_val_test(model.hw)

    # weights
    total = normal_len + carcinoma_len
    normal_ratio, carcinoma_ratio = total/normal_len, total/carcinoma_len
    
    weight_tensor = torch.tensor([carcinoma_ratio, normal_ratio], device=device)
    weight_dic = {0: carcinoma_ratio, 1: normal_ratio}
    train_sample_weights = compute_sample_weight(weight_dic, y_train) / (total)
    test_sample_weights = compute_sample_weight(weight_dic, y_test) / total
    
    # dataloaders
    train_loader, val_loader, test_loader = dt.create_dataloaders(x_train, y_train, x_val, y_val, x_test, y_test, train_sample_weights, batch_size=32)

    # configure weighted loss
    model.configure_loss_func(weight_tensor)

    model_dict = {
        "version": version,
        "model": model_name,
        "date": date_today,
        "numb_epochs": epochs,
        "accuracy": None,
        "balanced_accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0
    }

    y_values = {
        "y_true": None,
        "y_predict": None
    }

    train_results = {
        "train_accuracies": None,
        "val_accuracies": None,
        "train_losses": None,
        "val_losses": None
    }

    # train
    train_accuracies, train_losses, val_accuracies, val_losses, y_predict = tr.train(model, train_loader, val_loader, weight_tensor)

    print("Saving training data...", end="\n\n")
    
    # transforming into np.array
    array_train_accuracies = np.asarray(train_accuracies)
    array_train_losses = np.asarray(train_losses)
    array_val_accuracies = np.asarray(val_accuracies)
    array_val_losses = np.asarray(val_losses)

    train_results["train_accuracies"] = array_train_accuracies
    train_results["val_accuracies"] = array_val_accuracies
    train_results["train_losses"] = array_train_losses
    train_results["val_losses"] = array_val_losses
    utils.save_train_results(model, train_results)
    
    # test
    test_accuracy, y_predict, y_true = tr.test(model, test_loader)

    y_test_predict = []
    y_test_true = []
    for _, x in enumerate(y_predict):
        for y in x:
            y_test_predict.append(int(y))
            
    for _, x in enumerate(y_true):
        for y in x:
            y_test_true.append(int(y))

    y_test_predict = np.array(y_test_predict)
    y_test_true = np.array(y_test_true)

    print("Saving test data...")

    balanced_test_accuracy = metrics.balanced_accuracy_score(y_test_true, y_test_predict, sample_weight=test_sample_weights) * 100
    tn, fp, fn, tp = metrics.confusion_matrix(y_test_true, y_test_predict).ravel()
    precision = metrics.precision_score(y_test_true, y_test_predict)
    recall = metrics.recall_score(y_test_true, y_test_predict)
    f1 = metrics.f1_score(y_test_true, y_test_predict)
        
    model_dict["accuracy"] = test_accuracy   
    model_dict["balanced_accuracy"] = balanced_test_accuracy
    model_dict["tn"] = tn
    model_dict["fp"] = fp
    model_dict["fn"] = fn
    model_dict["tp"] = tp
    model_dict["precision"] = precision
    model_dict["recall"] = recall
    model_dict["f1"] = f1

    print("Balanced accuracy: ", balanced_test_accuracy)

    utils.save_test_results(model_dict)

if __name__ == "__main__":
    print("Starting Program!")
    print("*"*20)
    
    # Configure params
    arg = sys.argv
    epochs, lr, model_name, version, op, date_today = get_arg(arg)

    # Run train + test programs
    main(date_today, epochs, lr, op, model_name, version)

    print("*"*20)
    print("Program ended!")