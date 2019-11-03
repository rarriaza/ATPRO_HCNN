import argparse
import os
import datasets
import models
import logging
from datetime import datetime
from datasets.preprocess import train_test_split, shuffle_data


def get_model_directory():
    model_directory = args.model
    if model_directory == '':
        now = datetime.now()
        timestamp = now.strftime('%Y%m%d%H%M%S')
        model_directory = f'./saved_models/{args.name}/{timestamp}'
    os.makedirs(model_directory, exist_ok=True)
    return model_directory


def get_results_file():
    results_directory = os.path.dirname(args.results)
    if results_directory == '':
        now = datetime.now()
        timestamp = now.strftime('%Y%m%d%H%M%S')
        results_file = f'./results/{args.name}/{timestamp}.json'
        results_directory = os.path.dirname(results_file)
    os.makedirs(results_directory, exist_ok=True)
    return results_file


def get_data_directory(args):
    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def get_logs_directory():
    logs_dir = "./logs"
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def get_data(dataset, data_directory):
    if dataset == 'cifar100':
        logging.info('Getting CIFAR-100 dataset')
        tr, te, fine2coarse, n_fine, n_coarse = datasets.get_cifar100(
            data_directory)
        logging.debug(
            f'Training set: x_dims={tr[0].shape}, y_dims={tr[1].shape}')
        logging.debug(
            f'Testing set: x_dims={te[0].shape}, y_dims={te[1].shape}')
    return tr, te, fine2coarse, n_fine, n_coarse


def main(args):
    logging.basicConfig(level=args.log_level)

    logging.info('Creating directories')

    logs_directory = get_logs_directory()
    logging.debug(f'Logs directory: {logs_directory}')

    model_directory = get_model_directory()
    logging.debug(f'Models directory: {model_directory}')

    data_directory = get_data_directory(args)
    logging.debug(f'Data directory: {data_directory}')

    results_directory = get_results_directory()
    logging.debug(f'Results directory: {results_directory}')

    logging.info('Getting data')
    data = get_data(args.dataset, data_directory)
    training_data = data[0]
    testing_data = data[1]
    fine2coarse = data[2]
    n_fine_categories = data[3]
    n_coarse_categories = data[4]

    logging.info('Building model')
    net = models.HDCNNBaseline(n_fine_categories,
                               n_coarse_categories, logs_directory,
                               model_directory, args)

    if args.train:
        logging.info('Entering training')
        training_data = shuffle_data(training_data)
        training_data, validation_data = train_test_split(training_data)
        net.train_shared_layers(training_data)
        net.sync_parameters()
        net.train_coarse_classifier(
            training_data, validation_data, fine2coarse)
        net.train_fine_classifiers(
            training_data, validation_data, fine2coarse)
    if args.test:
        logging.info('Entering testing')
        net.predict(testing_data, fine2coarse, args.results_file)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='HDCNN baseline running script'
    )

    parser.add_argument('-tr', '--train', help='Train a new model',
                        action='store_true')
    parser.add_argument('-te', '--test', help='Test a model',
                        action='store_true')
    parser.add_argument('-m', '--model', help='Specify where to store model',
                        type=str, default='')
    parser.add_argument('-n', '--name', help='Model run name',
                        type=str, default='baseline_hdcnn')
    parser.add_argument('-d', '--dataset', help='Dataset to use',
                        type=str, default='cifar100',
                        choices=['cifar100'])
    parser.add_argument('--data_dir', help='Where to store data on the local'
                                           ' machine (defaults to ./data)',
                        type=str, default='./data')
    parser.add_argument('-l', '--log_level', help='Logs level',
                        type=str, default='INFO',
                        choices=['WARNING', 'INFO', 'DEBUG', 'ERROR'])
    parser.add_argument('-r', '--results', help='Results file',
                        type=str, default='')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    # TODO: remove
    args.train = True
    args.log_level = 'DEBUG'

    main(args)
