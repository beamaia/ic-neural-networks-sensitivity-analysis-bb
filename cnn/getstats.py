import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd

from sklearn import metrics, svm
from sklearn.utils.class_weight import compute_sample_weight

import sys

from glob import glob
import fnmatch

def get_paths(PATH):
    PATH_y_pred = PATH + "/y_predict/y_predict_*"
    PATH_y_true = PATH + "/y_predict/y_test_*"

    pred = glob(PATH_y_pred)
    true = glob(PATH_y_true)

    if len(pred) != len(true):
        print("y_predict is different than y_true!")
        sys.exit(-1)

    return pred, true

def get_info(pred_path, true_path, PATH):

    if PATH != ".":
        pred_path = pred_path.replace(PATH, ".")
        true_path = true_path.replace(PATH, ".")

    _, _, pred_string = pred_path.split("/")
    _, _, true_string = true_path.split("/")
    pred_string, _ = pred_string.split(".")
    true_string, _ = true_string.split(".")
    _, pred_string = pred_string.split("y_predict_")
    _, true_string = true_string.split("y_test_")
    pred_model, pred_date, pred_version = pred_string.split("_")
    true_model, true_date, true_version = true_string.split("_")
    
    same_model = pred_model == true_model
    same_date = pred_date == true_date
    same_version = pred_version == true_version

    if same_model and same_version and same_date:
        return pred_model, pred_date, pred_version
    else: 
        print("Paths are for different models!")
        sys.exit(-1)


def get_accuracy(df, model, date, version):
    same_everything = df.loc[(df["Model"]==model) & (df["Date"] == date) & (df["Version"] == int(version))]
    index_size = len(same_everything.index)

    if index_size is not 1:
        return False, False
    else:
        return same_everything["Accuracy"].values[0], same_everything["Set"].values[0]


def get_stats(PATH):
    # model, date, version, positive_class, negative_class = get_info()
    PATH_accuracies = PATH+"/test_accuracies/accuracies.csv"
    df_accuracies = pd.read_csv(PATH_accuracies, index_col=False)

    pred, true = get_paths(PATH)    
    pred.sort(), true.sort()
    
    models = []
    dates = []
    versions = []
    tn_list = []
    tp_list = []
    fn_list = []
    fp_list = []
    precisions = []
    recalls = []
    f1_scores = []
    # auc_scores = []
    accuracies = []
    b_accuracies = []
    sets = []

    for pred_path, true_path in zip(pred, true):
        model, date, version = get_info(pred_path, true_path, PATH)
        ac, set_used = get_accuracy(df_accuracies, model, date, version)

        if (not ac):
            continue

        y_pred = np.loadtxt(pred_path)
        y_true = np.loadtxt(true_path)
        accuracy = metrics.accuracy_score(y_true, y_pred)

        if float(ac) != float(accuracy) * 100:
            continue 

        accuracies.append(float(accuracy)*100)

        models.append(model)
        dates.append(date)
        versions.append(version)
        sets.append(set_used)
        unique, counts = np.unique(y_true, return_counts=True)
        class_weight = dict(zip(unique, counts))
        test_sample_weights = compute_sample_weight(class_weight, y_true)
        balanced_test_accuracy = metrics.balanced_accuracy_score(y_true, y_pred, sample_weight=test_sample_weights)

        b_accuracies.append(balanced_test_accuracy)
        # y_true1 = np.count_nonzero(y_true)
        # y_true0 = len(y_true) - y_true1
        # weights = compute_sample_weight({1:y_true1, 0: y_true0}, y_true)
  
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()

        tn_list.append(tn)
        tp_list.append(tp)
        fn_list.append(fn)
        fp_list.append(fp)

        precision = metrics.precision_score(y_true, y_pred)
        recall = metrics.recall_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred)

        # auc = metrics.roc_auc_score(y_true, y_pred, sample_weight=weights)

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        # auc_scores.append(auc)

    tn_list = np.array(tn_list)
    tp_list = np.array(tp_list)
    fn_list = np.array(fn_list)
    fp_list = np.array(fp_list)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1_scores = np.array(f1_scores)
    # auc_scores = np.array(auc_scores)
    accuracies = np.array(accuracies)
    b_accuracies = np.array(b_accuracies)
    sets = np.array(sets)

    dict_stats = {"Model": models, 
             "Date": dates,
             "Version": versions,
             "Set": sets,
             "Accuracy": accuracies, 
             "Balanced accuracy": b_accuracies,
             "Precision": precisions,
             "Recall": recalls,
             "F1": f1_scores}
            #  "AUC": auc_scores}

    dict_pos_neg = {"Model": models, 
             "Date": dates,
             "Version": versions,
             "Set": sets,
             "True Positive": tp_list,
             "False Negatives": fn_list,
             "False Positives": fp_list, 
             "True Negatives": tn_list}

    stats = pd.DataFrame(dict_stats)
    pos_neg = pd.DataFrame(dict_pos_neg)
    return stats, pos_neg

if __name__ == "__main__":
    args = sys.argv
    PATH = args[1]

    stats, pos_neg = get_stats(PATH)

    PATH_stats = PATH + "/stats/stats.csv"
    PATH_pos_neg = PATH + "/stats/positive_negatives_stats.csv"
    stats.to_csv(PATH_stats, index=False)
    pos_neg.to_csv(PATH_pos_neg, index=False)
