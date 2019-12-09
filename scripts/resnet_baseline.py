import argparse
import logging

import os

import models
from datasets.preprocess import train_test_split, shuffle_data
from scripts.resnet_attention import get_logs_file, get_model_directory, get_data_directory, get_results_file, get_data


def main(args):
    logs_file = get_logs_file(args.name)
    logs_directory = os.path.dirname(logs_file)

    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        filename=logs_file,
                        filemode='w')
    ch = logging.StreamHandler()
    ch.setLevel(args.log_level)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger = logging.getLogger('')
    logger.addHandler(ch)

    logger.debug(f'Logs file: {logs_file}')

    model_directory = get_model_directory(args)
    logger.debug(f'Models directory: {model_directory}')

    data_directory = get_data_directory(args)
    logger.debug(f'Data directory: {data_directory}')

    results_file = get_results_file(args)
    logger.debug(f'Results file: {results_file}')

    logger.info('Getting data')
    data = get_data(args.dataset, data_directory)
    training_data = data[0]
    testing_data = data[1]
    validation_data = data[2]
    fine2coarse = data[3]
    n_fine_categories = data[4]
    n_coarse_categories = data[5]
    input_shape = training_data[0][0].shape

    logger.info('Building model')
    net = models.ResNetBaseline(n_fine_categories=n_fine_categories,
                                n_coarse_categories=n_coarse_categories,
                                input_shape=input_shape,
                                logs_directory=logs_directory,
                                model_directory=model_directory,
                                args=args)

    if args.train:
        logger.info('Entering training')
        trdx, trdy, _ = shuffle_data(training_data)
        training_data = trdx, trdy
        net.train(training_data, validation_data)
    if args.test:
        logger.info('Entering testing')
        net.predict_fine(testing_data, results_file, fine2coarse)  # args.results)


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
