from training_classification import train as train_c
from training_classification import getModel as model_c
from training_segmentation import train as train_s
from training_segmentation import getModel as model_s
from quaternion_layers.utils import Params
import click
import numpy as np
np.random.seed(314)

@click.command()
@click.argument('task')
@click.option('--mode', default='quaternion', help='value type of model (real, complex, quaternion)')
@click.option('--num-blocks', '-nb', default=2, help='number of residual blocks per stage')
@click.option('--start-filters', '-sf', default=8, help='number of filters in first layer')
@click.option('--dropout', '-d', default=0, help='dropout percent')
@click.option('--batch-size', '-bs', default=8, help='batch size')
@click.option('--num-epochs', '-e', default=200, help='total number of epochs')
@click.option('--dataset', '-ds', default='cifar10', help='dataset to train and test on')
@click.option('--activation', '-act', default='relu', help='activation function to use')
@click.option('--initialization', '-init', default='quaternion', help='initialization scheme to use')
@click.option('--learning-rate', '-lr', default=1e-3, help='learning rate for optimizer')
@click.option('--momentum', '-mn', default=0.9, help='momentum for batch norm')
@click.option('--decay', '-dc', default=0, help='decay rate of optimizer')
@click.option('--clipnorm', '-cn', default=1.0, help='maximum gradient size')
def runner(task, mode, num_blocks, start_filters, dropout, batch_size, num_epochs, dataset, 
           activation, initialization, learning_rate, momentum, decay, clipnorm):
    
    param_dict = {"mode": mode,
                  "num_blocks": num_blocks,
                  "start_filter": start_filters,
                  "dropout": dropout,
                  "batch_size": batch_size,
                  "num_epochs": num_epochs,
                  "dataset": dataset,
                  "act": activation,
                  "init": initialization,
                  "lr": learning_rate,
                  "momentum": momentum,
                  "decay": decay,
                  "clipnorm": clipnorm
    }
    
    params = Params(param_dict)

    if task == 'classification':
        model = model_c(params)
        train_c(params, model)
    elif task == 'segmentation':
        model = model_s(params)
        train_s(params, model)
    else:
        print("Invalid task chosen...")


if __name__ == '__main__':
    runner()