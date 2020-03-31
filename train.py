import pickle
import tensorflow as tf
import numpy as np
from tools import tools, params_help, data
from network import models_att
import os
import argparse
import pprint

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description='train')

    # general
    parser.add_argument('--data-type', help='scale', required=True, choices=['scale'], type=str)
    parser.add_argument('--mode', help='dt_ft, gt', required=True, choices=['gt', 'dt_ft'], type=str)

    # optional arguments
    parser.add_argument('--test-indices', help='test idx ', type=str)
    parser.add_argument('--mask-type', help='mask type ', type=str)
    parser.add_argument('--graph', help='index of graphs', type=int, default=0)
    parser.add_argument('--knn', help='expand of neighbourhood', type=int)
    parser.add_argument('--layers', help='number of layers', type=int)
    parser.add_argument('--in-joints', help='number of input joints', type=int, default=17)
    parser.add_argument('--out-joints', help='number of output joints', type=int, default=17)
    parser.add_argument('--dropout', help='dropout probability', type=float)
    parser.add_argument('--channels', help='number of channels', type=int, default=64)

    parser.add_argument('--in-F', help='feature channels of input data', type=int, default=2, choices=[2, 3])
    parser.add_argument('--flip-data', help='train time flip', action='store_true')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    datareader = data.DataReader()
    train_data, test_data = datareader.read_2d(which=args.data_type, mode=args.mode, read_confidence=True if args.in_F == 3 else False)
    train_labels, test_labels = datareader.read_3d(which=args.data_type, mode=args.mode)

    if args.flip_data:
        # only work for scale 
        train_data = data.flip_data(train_data)
        train_labels = data.flip_data(train_labels)
    
    # params
    params = params_help.get_params(is_training=True, gt_dataset=train_labels)
    params_help.update_parameters(args, params)
    print(pprint.pformat(params))

    network = models_att.cgcnn(**params)
    network.fit(train_data, train_labels, test_data, test_labels)

if __name__ == '__main__':
    main()

