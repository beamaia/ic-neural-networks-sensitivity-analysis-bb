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
    set_numb = 3
    epochs = 200
    lr = 0.0001
    model_name = "resnet50"
    dp = 0.5
    version = 1
    op = "Adam"
    loss_func = "CrossEntropy"

    for arg in args:
        if arg == "main.py":
            continue
        
        param, value = arg.split("=")
        if param == "set" or param == "set_num":
            set_numb = int(value)
        elif param == "epoch" or param == "epochs" or param == "num_epochs":
            epochs = int(value)
        elif param == "lr" or param == "learning_rate":
            lr = float(value)
        elif param == "model" or param == "model_name":
            model_name = value
        elif param == "version":
            version = int(value)
        elif param == "dp" or param == "dropout":
            dp = float(value)
        elif param == "op":
            op = param

    today = date.today()
    date_today = today.strftime("%d-%m-%y")

    return set_numb, epochs, lr, model_name, dp, version, op, loss_func, date_today

def main(date_today, set_numb, epochs, lr, op, model_name, dp, version):
    model = md.Model(epochs, lr, op, model_name, dp, version)

    # images    
    images_normal, images_carcinoma = dt.image_paths(set_numb)
    x_normal = dt.process_images(images_normal)
    x_carcinoma = dt.process_images(images_carcinoma)
    images, labels = dt.create_images_labels(x_normal, x_carcinoma, patch_size=model.hw)

    normal_len, carcinoma_len = len(x_normal), len(x_carcinoma)
    total = normal_len + carcinoma_len
    normal_ratio, carcinoma_ratio = normal_len/total, carcinoma_len/total
    
    weight_tensor = torch.tensor([carcinoma_ratio, normal_ratio])
    weight_dic = {0: carcinoma_ratio, 1: normal_ratio}
    
    # split
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = dt.create_train_test(images, labels, 0.2)

    train_sample_weights = compute_sample_weight(weight_dic, y_train) / (len(y_test) + len(y_train) + len(y_val))
    val_sample_weights = compute_sample_weight(weight_dic, y_val) / (len(y_test) + len(y_train) + len(y_val))
    test_sample_weights = compute_sample_weight(weight_dic, y_test) / (len(y_test) + len(y_train) + len(y_val))

    # dataloaders
    train_loader, val_loader, test_loader = dt.create_dataloaders(x_train, y_train, x_val, y_val, x_test, y_test, train_sample_weights, val_sample_weights, test_sample_weights, batch_size=32)

    model_dict = {
        "model": model_name,
        "date": date_today,
        "set_numb": set_numb,
        "numb_epochs": epochs,
        "accuracy": None,
        "balanced_accuracy": None,
        "precision": None,
        "recall": None
    }

    y_values = {
        "y_true": None,
        "y_predict": None
    }

    # train
    train_accuracies, train_losses, val_accuracies, val_losses, y_predict = tr.train(model, model_name, train_loader, val_loader, weight_tensor, **params)

    print("Saving training data...", end="\n\n")
    
    # transforming into np.array
    array_train_accuracies = np.asarray(train_accuracies)
    array_train_losses = np.asarray(train_losses)
    array_val_accuracies = np.asarray(val_accuracies)
    array_val_losses = np.asarray(val_losses)

    # save_model_dict (model, model_name,  date_today, version)

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

    balanced_test_accuracy = metrics.balanced_accuracy_score(y_test_true, y_test_predict, sample_weight=test_sample_weights)
    model_dict["accuracy"] = test_accuracy   
    model_dict["balanced_accuracy"] = balanced_test_accuracy
    y_values["y_test"] = y_test_true
    y_values["y_predict"] = y_test_predict

    print("Balanced accuracy: ", balanced_test_accuracy)

    utils.save_test_accuracy(test_accuracy, balanced_test_accuracy, model_name, date_today, version, new=False, sets=set_numb) # change here
    utils.save_y_true_predict(y_test_true, y_test_predict, model_name, date_today, version)

    # Saving info
    utils.save_txt_accuracy_loss(array_train_accuracies, array_train_losses, date_today, model_name, version, training=True)
    
    print("Plotting images...")
    utils.plot_accuracy_loss(epochs, model_name, losses=array_train_losses, accuracies=array_train_accuracies, date=date_today, version=version, training=True)

    # utils.save_txt_accuracy_loss(array_val_accuracies, array_val_losses, date, model_name, version, training=False)
    utils.plot_accuracy_loss(epochs, model_name, losses=array_val_losses, accuracies=array_val_accuracies, date=date_today, version=version, training=False)


if __name__ == "__main__":
    print("Starting Program!")
    
    # Configure params
    arg = sys.argv
    set_numb, epochs, lr, model_name, dp, version, op, date_today = get_arg(arg)

    # Run train + test programs
    main(date_today, set_numb, epochs, lr, op, model_name, dp, version)
    
    print("Program ended!")