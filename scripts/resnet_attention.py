import argparse
import logging
from datetime import datetime

import os

import datasets
import models
from datasets.preprocess import train_test_split, shuffle_data


def get_model_directory(args):
    model_directory = args.model
    if model_directory == '':
        model_directory = f'./saved_models/{args.name}'
    os.makedirs(model_directory, exist_ok=True)
    return model_directory


def get_results_file(args):
    results_directory = os.path.dirname(args.results)
    if results_directory == '':
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d-%H%M%S")
        results_file = f'./results/{args.name}/{timestamp}.json'
        results_directory = os.path.dirname(results_file)
    os.makedirs(results_directory, exist_ok=True)
    return results_file


def get_data_directory(args):
    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def get_logs_file(model_name):
    logs_dir = "./logs"
    os.makedirs(logs_dir, exist_ok=True)
    logs_file = os.path.join(logs_dir, f"{model_name}.log")
    return logs_file


def get_data(dataset, data_directory):
    if dataset == 'cifar100':
        logging.info('Getting CIFAR-100 dataset')
        tr, te, fine2coarse, n_fine, n_coarse = datasets.get_cifar100(
            data_directory)
        tr_x, tr_y, _ = shuffle_data(tr, random_state=0)
        tr = tr_x, tr_y
        tr, val = train_test_split(tr)
        logging.debug(
            f'Training set: x_dims={tr[0].shape}, y_dims={tr[1].shape}')
        logging.debug(
            f'Testing set: x_dims={te[0].shape}, y_dims={te[1].shape}')
    return tr, te, val, fine2coarse, n_fine, n_coarse


def main(args):
    logs_file = get_logs_file(args.name)
    logs_directory = os.path.dirname(logs_file)

    logging.basicConfig(level=logging.DEBUG,
                        filename=logs_file,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        filemode='a')
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

    if args.debug_mode:
        logger.info('Removing data. Keeping only 100 samples')
        training_data = training_data[0][:100], training_data[1][:100]
        testing_data = testing_data[0][:100], testing_data[1][:100]
        validation_data = validation_data[0][:100], validation_data[1][:100]

    fine2coarse = data[3]
    n_fine_categories = data[4]
    n_coarse_categories = data[5]
    input_shape = training_data[0][0].shape

    best_cc = None
    best_fc = None

    logger.info('Building model')
    net = models.ResNetAttention(n_fine_categories=n_fine_categories,
                                 n_coarse_categories=n_coarse_categories,
                                 input_shape=input_shape,
                                 logs_directory=logs_directory,
                                 model_directory=model_directory,
                                 args=args)

    if args.train_c:
        logger.info('Entering Coarse Classifier training')
        best_cc = net.train_coarse(training_data, validation_data, fine2coarse)
    if args.train_f:
        logger.info('Entering Fine Classifier training')
        best_fc = net.train_fine(training_data, validation_data, fine2coarse)
    if args.train_full:
        logger.info('Entering Full Classifier training')
        best_fc = net.train_both(training_data, validation_data, fine2coarse)
    if args.test_full:
        logger.info('Entering testing')
        net.predict_full(testing_data, fine2coarse, results_file)
    if args.test:
        logger.info('Entering testing')

        # Nice-to-have: maybe there is a better way of doing this loading thing
        if args.load_model_cc is not None:
            net.load_cc_model(args.load_model_cc)
        elif best_cc is not None:
            net.load_cc_model(best_cc)
        else:
            net.load_best_cc_model()
        yc_pred = net.predict_coarse(testing_data, fine2coarse, results_file)

        x_test_feat = net.get_feature_input_for_fc(testing_data[0])
        testing_data = x_test_feat, yc_pred, testing_data[1]

        # Nice-to-have: maybe there is a better way of doing this loading thing
        if args.load_model_fc is not None:
            net.load_fc_model(args.load_model_fc)
        elif best_fc is not None:
            net.load_fc_model(best_fc)
        else:
            net.load_best_fc_model()
        net.predict_fine(testing_data, results_file)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='ResNet baseline running script'
    )
    parser.add_argument('-debug', '--debug_mode', help='Train in one epoch with few samples',
                        action='store_true')
    parser.add_argument('-tr_c', '--train_c', help='Train the coarse classifier',
                        action='store_true')
    parser.add_argument('-tr_f', '--train_f', help='Train the fine classifier',
                        action='store_true')
    parser.add_argument('-tr_full', '--train_full', help='Train the full classifier',
                        action='store_true')
    parser.add_argument('-te', '--test', help='Test a model',
                        action='store_true')
    parser.add_argument('-te_full', '--test_full', help='Test a full model',
                        action='store_true')
    parser.add_argument('-m', '--model', help='Specify where to store model',
                        type=str, default='')
    parser.add_argument('-n', '--name', help='Model run name',
                        type=str, default='attention')
    parser.add_argument('-d', '--dataset', help='Dataset to use',
                        type=str, default='cifar100',
                        choices=['cifar100'])
    parser.add_argument('--data_dir', help='Where to store data on the local'
                                           ' machine (defaults to ./data)',
                        type=str, default='./data')
    parser.add_argument('--load_model_cc', help='Load pre trained cc model',
                        type=str, default=None)
    parser.add_argument('--load_model_fc', help='Load pre trained fc model',
                        type=str, default=None)
    parser.add_argument('-l', '--log_level', help='Logs level',
                        type=str, default='INFO',
                        choices=['WARNING', 'INFO', 'DEBUG', 'ERROR'])
    parser.add_argument('-r', '--results', help='Results file',
                        type=str, default='')

    return parser.parse_args()


if __name__ == '__main__':
    main(args)
