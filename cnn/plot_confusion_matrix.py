import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

def plot_confusion_matrix(array, cnn, optim, version):
    df_cm = pd.DataFrame(array, range(2), range(2))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g') # font size
    
    tick_marks = np.arange(2)
    target_names = ["True", "False"]
    plt.xticks(tick_marks +0.5, target_names, va='center')
    plt.yticks(tick_marks+0.5, target_names, rotation=360, va='center')
    
    plt.xlabel("True Class")
    plt.ylabel("Predicted Class")
    
    plt.title("Confusion Matrix - " + cnn + " - " + optim)
    path = f'results/confusion_matrix/{cnn}-{optim}-v{version}-confmatrix.png'
    
    plt.savefig(path, facecolor="w", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    raw_data = pd.read_csv("results/metrics.csv")
    results = raw_data.drop([i for i in range(12)])

    for index, df_index in enumerate(results.index):
        tp = results.tp[df_index]
        fp = results.fp[df_index]
        tn = results.tn[df_index]
        fn = results.fn[df_index]
        version = results.version[df_index]

        array = np.array([[tp, fp], [fn, tn]])
        
        cnn = results.model[df_index]
        optim = results.optim[df_index]
        
        plot_confusion_matrix(array, cnn, optim, version)