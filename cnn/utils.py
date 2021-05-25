# importing libraries
import torch
import torchvision
from torchvision import models
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

def save_txt_accuracy_loss (accuracies, losses, date, model, version=1, training=True):
    if training:
        PATH_accuracy = "./train_accuracies/train_accuracies_"
        PATH_loss = "./train_losses/train_losses_"
    else:
        PATH_accuracy = "./val_accuracies/val_accuracies_"
        PATH_loss = "./val_losses/val_losses_"
    
    PATH =  str(model) + "_" + str(date) + "_" + str(version) + ".txt"
    PATH_accuracy += PATH
    PATH_loss += PATH

    np.savetxt(PATH_accuracy, accuracies)
    np.savetxt(PATH_loss, losses)

def save_model_dict (model, model_name, date, version=1):
    PATH = './saved_models/' + str(model_name)  + "_" + str(date) + "_" + str(version)
    torch.save(model.state_dict(), PATH)

def load_model_dict (model_name, date, version=1, cpu=False):
    PATH = './saved_models/' + str(model_name)  + "_" + str(date) + "_" + str(version)
    print(f"CPU: {cpu}",end="\n\n")

    if model_name is "resnet50":
        model = models.resnet50(pretrained=True)

    if not cpu:
        model.load_state_dict(torch.load(PATH))
    else:
        device = torch.device('cpu')
        model.load_state_dict(torch.load(PATH, map_location=device))
    return model

def calculate_true_false_results (y_test, y_predict):
    y_predict = np.array(y_predict)
    positive_pred = y_predict == 1
    negative_pred = y_predict == 0  

    positive = np.array(y_test) == 1
    negative = np.array(y_test) == 0

    masks = pd.DataFrame({'Positive':positive, 
                          'Negative': negative, 
                          'Predicted Positive': positive_pred, 
                          'Predicted Negative': negative_pred})

    true_positive_mask = (positive & positive_pred)
    true_negative_mask = (negative & negative_pred)
    false_positive_mask = positive_pred & negative
    false_negative_mask = negative_pred & positive
    
    fp = false_positive_mask.sum()
    fn = false_negative_mask.sum()
    tp = true_positive_mask.sum()
    tn = true_negative_mask.sum()

    # True Positive - False Positive
    # False Negative - True Negative
    df_amount = pd.DataFrame([[tp, fp], [fn, tn]])
 
    df_amount.index.name = 'Test'
    df_amount.index.name = 'Disease'
    df_amount.columns = ["+", "-"]
    df_amount = df_amount.set_index(pd.Index(["+", "-"]))

    df_percentage = pd.DataFrame([[tp, fp], [fn, tn]])
    df_percentage.index.name = 'Test'
    df_percentage.index.name = 'Disease'
    df_percentage.columns = ["+", "-"]
    df_percentage = df_percentage.set_index(pd.Index(["+", "-"]))
    df_percentage = df_percentage / y_test.shape[0] * 100   

    return df_amount, df_percentage, masks


def save_dataframes (y_test, y_predict, model_name, date, version=1):
    # y_predict = []
    # for _, x in enumerate(y_test_predict):
    #     for y in x:
    #         y_predict.append(int(y))

    df_amount, df_percentage, masks = calculate_true_false_results(y_test, y_predict)

    PATH = str(model_name) + "_" + str(date) + "_" + str(version)
    PATH_amount = "./dataframes/TP-FN_amount_" + PATH + ".csv"
    PATH_percentage = "./dataframes/TP-FN_percentage_" + PATH + ".csv"
    PATH_yt = "./y_predict/y_test_" + PATH + ".txt"
    PATH_yp = "./y_predict/y_predict_" + PATH + ".txt"
    PATH_mask = "./y_predict/" + PATH + ".csv"

    df_amount.to_csv(PATH_amount,index=True)
    df_percentage.to_csv(PATH_percentage,index=True)
    masks.to_csv(PATH_mask, index=True)

    np.savetxt(PATH_yp, y_predict, delimiter=",")
    np.savetxt(PATH_yt,y_test, delimiter=",")

def save_train_results (model, results):
    PATH = "./results/train_val/" + str(model.model_name) + "-" + str(model.op_name) + "-v" + str(model.version) + ".csv"
    df = pd.DataFrame(results)

    df.to_csv(PATH, index=False)

def save_test_results (results):
    PATH = "./results/metrics.csv"
    df = pd.DataFrame([results])
    
    df_origin = pd.read_csv(PATH, index_col=False)
    df = df_origin.append(df)
    df.to_csv(PATH, index=False)
