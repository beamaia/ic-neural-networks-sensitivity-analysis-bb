import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

def get_df(PATH):
    stats = pd.read_csv(PATH, sep=",")
    stats = stats.drop(["Unnamed: 0", "Version"], axis=1)
    df_dates = stats["Date"].str.split("-")
    dates = []

    for string in df_dates:
        dates.append(int(string[0]))

    stats.insert(loc=0, column='Day', value=dates)
    stats = stats.drop(["Date"], axis=1)  

    return stats

def get_summary(df):
    return df.loc[(df["Set"] == 1)].describe(), df.loc[(df["Set"] == 2)].describe(), df.loc[(df["Set"] == 3)].describe()

def get_accuracy(summary):
    mean = summary.iloc[1]["Accuracy"]
    mins = summary.iloc[3]["Accuracy"]
    maxs = summary.iloc[7]["Accuracy"]

    return mean, mins, maxs

def get_recall(summary):
    mean = summary.iloc[1]["Recall"]
    mins = summary.iloc[3]["Recall"]
    maxs = summary.iloc[7]["Recall"]

    return mean, mins, maxs

def plot_min_mean_max(values, version=1, set_num = 1, ac=True, strs=["resnet50", "mobilenet-v2", "inception-v3"], bar_num=3, extra="", PATH="./stats/"):
    labels = ['Min', 'Mean', 'Max']
    x = np.arange(len(labels))  
    fig, ax = plt.subplots(figsize=(10, 5))
    path = PATH
   
    def subcategorybar(labels, values, x, strings, width=0.8):
        color=['springgreen', 'dodgerblue', 'mediumpurple']
        # color=['royalblue', 'limegreen', 'darkorange']

        n = len(values)
        rects = []
        for i in range(n):
            a = plt.bar(x - width/2. + i/float(n)*width, values[i], width=width/float(n), align="edge", 
                        label=strings[i], color=color[i])   
            rects.append(a)
        plt.xticks(x, labels)
    
        return rects
    
    rects = subcategorybar(labels, values, x, strs)
    
    
    if set_num == 1:
        set_string = "Set 1"
    elif set_num == 2:
        set_string = "Set 2"
    elif set_num == 3:
        set_string = "Both Sets"
        
    if len(extra) != 0:
        set_string += " - " + extra
        path += extra.lower() + "_"

    if ac:
        add = "Accuracy"
    else:
        add = "Recall"
        
    title = 'Min, Mean and Max ' + add + " - "
    title = title + set_string
    ax.set_ylabel('Percentage (%)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper right',  bbox_to_anchor=(1.20, 1))

    if not ac:
        ax.set_ylim([0,1.1])
    else:
        ax.set_ylim([0,100])

    def autolabel(rects):
        for bar in rect:
            height = bar.get_height().round(2)
            ax.annotate('{}'.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 10),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    for rect in rects:
        autolabel(rect)

    fig.tight_layout()

    path += add.lower() + "_min_mean_max_set" + str(set_num) + "_version_" + str(version) + ".png"        
    plt.savefig(path, format="png", transparent=False, edgecolor="white")
