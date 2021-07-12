from numpy.core.numeric import True_
import pandas as pd
import matplotlib.pyplot as plt
import sys 

SAVE_PATH = 'results/train_val/vgg16-v12.1-dp=0.1'
DATA_PATH = 'results/train_val/vgg16-sgd-v12.1.csv'
LOSS = True
ACCURACY = True

def plot_aux(x, y_train, y_val):
    plt.plot(x, y_train, 'g', label='train')
    plt.plot(x, y_val, 'b', label='validation')
    plt.xlabel('Epoch')
    plt.legend()

def plot(raw_data, path=SAVE_PATH, losses=True, accuracy=False):
    tr_losses = raw_data.train_losses
    v_losses = raw_data.val_losses
    tr_accur = raw_data.train_accuracies
    v_accur = raw_data.val_accuracies
    epochs = range(200)

    if losses:
        plot_aux(epochs, tr_losses, v_losses)
        plt.ylabel('Average Loss')
        plt.title("Average Loss of Train and Validation per Epoch")
        plt.ylim(0, .5)
        plt.savefig(path + "-loss.png")

    if accuracy:
        plt.close()
        plot_aux(epochs, tr_accur, v_accur)
        plt.ylabel('Average Accuracy')
        plt.title("Average Accuracy of Train and Validation per Epoch")
        plt.ylim(0, 100)
        plt.savefig(path + "-accuracy.png")


if __name__ == '__main__':
    args = sys.argv
    
    data_path = DATA_PATH
    image_path = SAVE_PATH
    loss = LOSS
    accuracy = ACCURACY

    for arg in args:
        if arg == "plot_graph.py":
                continue      

        try:     
            param, value = arg.split("=")
        except ValueError:
            break
        
        if param == "data":
            data_path = value
        elif param == "img":
            image_path = value

    raw_data = pd.read_csv(data_path)
    plot(raw_data, image_path,)