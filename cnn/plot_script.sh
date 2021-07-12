echo "Starting ploting"
echo "DenseNet121"
python3 plot_graph.py data=results/train_val/densenet121-adam-v14.1.csv img=results/train_val/densenet121-v14.1-dp-0.1
python3 plot_graph.py data=results/train_val/densenet121-adam-v14.2.csv img=results/train_val/densenet121-v14.2-dp-0.1
python3 plot_graph.py data=results/train_val/densenet121-adam-v14.3.csv img=results/train_val/densenet121-v14.3-dp-0.1
python3 plot_graph.py data=results/train_val/densenet121-sgd-v18.1.csv img=results/train_val/densenet121-v18.1-dp-0.1
python3 plot_graph.py data=results/train_val/densenet121-sgd-v18.2.csv img=results/train_val/densenet121-v18.2-dp-0.1
python3 plot_graph.py data=results/train_val/densenet121-sgd-v18.3.csv img=results/train_val/densenet121-v18.3-dp-0.1

echo "Mobilenet-v2"
python3 plot_graph.py data=results/train_val/mobilenetv2-adam-v11.1.csv img=results/train_val/mobilenetv2-v11.1-dp-0.1
python3 plot_graph.py data=results/train_val/mobilenetv2-adam-v11.2.csv img=results/train_val/mobilenetv2-v11.2-dp-0.1
python3 plot_graph.py data=results/train_val/mobilenetv2-adam-v11.3.csv img=results/train_val/mobilenetv2-v11.3-dp-0.1
python3 plot_graph.py data=results/train_val/mobilenetv2-sgd-v17.1.csv img=results/train_val/mobilenetv2-v17.1-dp-0.1
python3 plot_graph.py data=results/train_val/mobilenetv2-sgd-v17.2.csv img=results/train_val/mobilenetv2-v17.2-dp-0.1
python3 plot_graph.py data=results/train_val/mobilenetv2-sgd-v17.3.csv img=results/train_val/mobilenetv2-v17.3-dp-0.1

echo "ResNet 50"
python3 plot_graph.py data=results/train_val/resnet50-adam-v10.4.csv img=results/train_val/resnet50-v10.4-dp-0.1
python3 plot_graph.py data=results/train_val/resnet50-adam-v10.2.csv img=results/train_val/resnet50-v10.2-dp-0.1
python3 plot_graph.py data=results/train_val/resnet50-adam-v10.3.csv img=results/train_val/resnet50-v10.3-dp-0.1
python3 plot_graph.py data=results/train_val/resnet50-sgd-v16.1.csv img=results/train_val/resnet50-v16.1-dp-0.1
python3 plot_graph.py data=results/train_val/resnet50-sgd-v16.2.csv img=results/train_val/resnet50-v16.2-dp-0.1
python3 plot_graph.py data=results/train_val/resnet50-sgd-v16.3.csv img=results/train_val/resnet50-v16.3-dp-0.1

echo "VGG16"
python3 plot_graph.py data=results/train_val/vgg16-adam-v13.1.csv img=results/train_val/vgg16-v13.1-dp-0.1
python3 plot_graph.py data=results/train_val/vgg16-adam-v13.2.csv img=results/train_val/vgg16-v13.2-dp-0.1
python3 plot_graph.py data=results/train_val/vgg16-adam-v13.3.csv img=results/train_val/vgg16-v13.3-dp-0.1
python3 plot_graph.py data=results/train_val/vgg16-sgd-v12.1.csv img=results/train_val/vgg16-v12.1-dp-0.1
python3 plot_graph.py data=results/train_val/vgg16-sgd-v12.2.csv img=results/train_val/vgg16-v12.2-dp-0.1
python3 plot_graph.py data=results/train_val/vgg16-sgd-v12.3.csv img=results/train_val/vgg16-v12.3-dp-0.1


echo "Finished ploting"
# SAVE_PATH = 'results/train_val/vgg16-v12.1-dp=0.1'
# DATA_PATH = 'results/train_val/vgg16-sgd-v12.1.csv'