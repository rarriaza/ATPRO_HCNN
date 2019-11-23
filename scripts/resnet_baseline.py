import argparse
import logging
from datetime import datetime

import os

import datasets
import models
from datasets.preprocess import train_test_split, shuffle_data


def get_model_directory():
    model_directory = args.model
    if model_directory == '':
        now = datetime.now()
        timestamp = now.strftime('%Y%m%d%H%M%S')
        model_directory = f'./saved_models/{args.name}'
    os.makedirs(model_directory, exist_ok=True)
    models_prefix = model_directory + f'/{timestamp}'
    return models_prefix


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


def get_logs_file():
    logs_dir = "./logs"
    os.makedirs(logs_dir, exist_ok=True)
    now = datetime.now()
    timestamp = now.strftime('%Y%m%d%H%M%S')
    logs_file = os.path.join(logs_dir, timestamp + ".log")
    return logs_file


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
    logs_file = get_logs_file()
    logs_directory = os.path.dirname(logs_file)

    logging.basicConfig(level=logging.DEBUG,
                        filename=logs_file,
                        filemode='w')
    ch = logging.StreamHandler()
    ch.setLevel(args.log_level)
    ch.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger = logging.getLogger('')
    logger.addHandler(ch)

    logger.debug(f'Logs file: {logs_file}')

    model_directory = get_model_directory()
    logger.debug(f'Models directory: {model_directory}')

    data_directory = get_data_directory(args)
    logger.debug(f'Data directory: {data_directory}')

    results_file = get_results_file()
    logger.debug(f'Results file: {results_file}')

    logger.info('Getting data')
    data = get_data(args.dataset, data_directory)
    training_data = data[0]
    testing_data = data[1]
    fine2coarse = data[2]
    n_fine_categories = data[3]
    n_coarse_categories = data[4]
    input_shape = training_data[0][0].shape

    logger.info('Building model')
    net = models.ResNetBaseline(n_fine_categories=n_fine_categories,
                                n_coarse_categories=n_coarse_categories,
                                input_shape=input_shape,
                                logs_directory=logs_directory,
                                model_directory=model_directory,
                                args=args)

    if args.load_model is not None:
        logger.info(f'Loading weights from {args.load_model}')
        net.load_models(args.load_model)

    if args.train:
        logger.info('Entering training')
        training_data = shuffle_data(training_data)
        training_data, validation_data = train_test_split(training_data)
        net.train(training_data, validation_data)
    if args.test:
        logger.info('Entering testing')
        net.predict_fine(testing_data, results_file)  # args.results)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='ResNet baseline running script'
    )

    parser.add_argument('-tr', '--train', help='Train a new model',
                        action='store_true')
    parser.add_argument('-te', '--test', help='Test a model',
                        action='store_true')
    parser.add_argument('-m', '--model', help='Specify where to store model',
                        type=str, default='')
    parser.add_argument('-n', '--name', help='Model run name',
                        type=str, default='baseline_resnet')
    parser.add_argument('-d', '--dataset', help='Dataset to use',
                        type=str, default='cifar100',
                        choices=['cifar100'])
    parser.add_argument('--data_dir', help='Where to store data on the local'
                                           ' machine (defaults to ./data)',
                        type=str, default='./data')
    parser.add_argument('--load_model', help='Load pre trained model',
                        type=str, default=None)
    parser.add_argument('-l', '--log_level', help='Logs level',
                        type=str, default='INFO',
                        choices=['WARNING', 'INFO', 'DEBUG', 'ERROR'])
    parser.add_argument('-r', '--results', help='Results file',
                        type=str, default='')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
