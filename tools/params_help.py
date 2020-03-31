import numpy as np
import scipy
import os, sys
import pickle, h5py
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_DIR)
import filter_hub
from random_graph import generate_random_graph 


def get_neighbour_matrix_by_hand(neighbour_dict, knn=1):
    assert len(neighbour_dict) == 17
    neighbour_matrix = np.zeros((17, 17), dtype=np.float32)
    for idx in range(len(neighbour_dict)):
        neigbour = [idx] + neighbour_dict[idx]
        neighbour_matrix[idx, neigbour] = 1
    if knn >= 2:
        neighbour_matrix = np.linalg.matrix_power(neighbour_matrix, knn)
        neighbour_matrix = np.array(neighbour_matrix!=0, dtype=np.float32)
    return neighbour_matrix


def get_laplacian_matrix(neighbour_dict, data, normalized=True, rescale=True):
    """
    Data: [N, 17*3]
    """
    # select out nonredundent joint pair and calculate mean length of limb
    data = np.reshape(data, (-1, 17, 3))
    assert len(neighbour_dict) == 17
    pair_list = []
    for idx in range(len(neighbour_dict)):
        for neighbour in neighbour_dict[idx]:
            if (idx, neighbour) not in pair_list and (neighbour, idx) not in pair_list:
               pair_list.append((idx, neighbour))
    limb_length_list = []
    for pair in pair_list:
        limb = np.mean(np.sqrt(np.sum(np.square(data[:, pair[0], :] - data[:, pair[1], :]), axis=1)))
        limb_length_list.append(limb)
    limb_length_array = np.array(limb_length_list, dtype=np.float32)

    # adjacency matrix
    adjacency_matrix = np.zeros((17, 17), dtype=np.float32)
    normed_length = np.exp(-np.square(limb_length_array/(np.mean(limb_length_array)+np.std(limb_length_array))))
    for pair, length in zip(pair_list, normed_length):
        adjacency_matrix[pair[0], pair[1]] = length
    adjacency_matrix += adjacency_matrix.T
    # No self-connections
    for idx in range(len(adjacency_matrix)):
        adjacency_matrix[idx, idx] = 0
    # Non-directed graph.
    assert np.abs(adjacency_matrix - adjacency_matrix.T).mean() < 1e-10

    # Laplacian matrix
    d = adjacency_matrix.sum(axis=0)
    assert d.ndim == 1
    if not normalized:
        D = np.diag(d, 0)
        L = D - adjacency_matrix
    else:
        d += np.spacing(np.array(0, adjacency_matrix.dtype))
        d = 1 / np.sqrt(d)
        D = np.diag(d, 0)
        I = np.identity(d.size, dtype=adjacency_matrix.dtype)
        L = I - D @ adjacency_matrix @ D

    # rescale Laplacian
    if rescale:
        if normalized:
            lmax = 2
        else:
            lmax = scipy.sparse.linalg.eigsh(
                    L, k=1, which='LM', return_eigenvectors=False)[0]
        I = np.identity(L.shape[0], dtype=L.dtype)
        L /= lmax / 2
        L -= I

    return L


def gen_neighbour_matrix_from_edges(edges, knn):
    neighbour_matrix = np.zeros((17, 17), dtype=np.float32)
    for idx in range(17):
        neighbour_matrix[idx, idx] = 1
    for pair in edges:
        neighbour_matrix[pair[0], pair[1]] = 1
    neighbour_matrix = neighbour_matrix + neighbour_matrix.T
    if knn >= 2:
        neighbour_matrix = np.linalg.matrix_power(neighbour_matrix, knn)

    neighbour_matrix = np.array(neighbour_matrix!=0, dtype=np.float32)
    return neighbour_matrix


def update_parameters(args, params):
    if args.test_indices:
        params['dir_name'] = 'test' + args.test_indices + '/'
    if args.knn:
        params['neighbour_matrix'] = get_neighbour_matrix_by_hand(filter_hub.neighbour_dict_set[args.graph], knn=args.knn)
    if args.layers is not None:
        params['num_layers'] = args.layers
    if args.in_joints:
        params['in_joints'] = args.in_joints
    if args.out_joints:
        params['out_joints'] = args.out_joints
    if args.dropout is not None:
        params['dropout'] = args.dropout
    if hasattr(args, 'channels') and args.channels:
        params['F'] = args.channels
    if hasattr(args, 'checkpoints') and args.checkpoints:
        params['checkpoints'] = args.checkpoints

    if args.in_F:
        params['in_F'] = args.in_F

    print(params['dir_name'])


def get_params(is_training, gt_dataset):

    params = {}
    params['dir_name'] = 'test1/'
    params['num_epochs'] = 200
    params['batch_size'] = 200
    # decay_strategy: lr * decay_rate ^ (epoch_num)
    params['decay_type'] = 'exp'  # 'step', 'exp'
    params['decay_params'] = {'decay_steps': 32000, 'decay_rate':0.96}  # param for exponential decay optimizer
    params['decay_params'].update({'boundaries': [250000, 500000, 1000000, 1350000], 'lr_values': [1e-3, 7e-4, 4e-4, 2e-4, 1e-4]})  # param for step optimizer
    params['eval_frequency'] = int(len(gt_dataset) / params['batch_size'])  # eval, summ & save after each epoch

    params['F'] = 64
    """
    mask_type:
        locally_connected
        locally_connected_learn
    """
    params['mask_type'] = 'locally_connected'
    params['init_type'] = 'random'  # same, ones, random; only used when learnable
    params['neighbour_matrix'] = get_neighbour_matrix_by_hand(filter_hub.neighbour_dict_set[0], knn=3)
    # import random
    # random.seed(146)
    # graph = generate_random_graph(17, 20)
    # params['neighbour_matrix'] = gen_neighbour_matrix_from_edges(graph.edges, knn=2)

    # # norm 1
    # params['neighbour_matrix'] = params['neighbour_matrix'] / np.sum(params['neighbour_matrix'], axis=1, keepdims=True)
    # # norm2
    # degree_matrix = np.diag(1/np.sqrt(np.sum(params['neighbour_matrix'], axis=1)))
    # params['neighbour_matrix'] = degree_matrix @ params['neighbour_matrix'] @ degree_matrix

    # params['neighbour_matrix'] = get_laplacian_matrix(filter_hub.neighbour_dict_set[0], gt_dataset,
    #     normalized=True, rescale=True)
    # params['neighbour_matrix'] = np.ones((17, 17))
    params['in_joints'] = 17
    params['out_joints'] = 17
    params['num_layers'] = 3
    params['in_F'] = 2
    params['residual'] = True
    params['max_norm'] = True
    params['batch_norm'] = True

    params['regularization'] = 0  # 5e-4, 0.0
    params['dropout'] = 0.25 if is_training else 0  # drop prob
    params['learning_rate'] = 1e-3
    params['checkpoints'] = 'final'

    return params


if __name__ == '__main__':
    import random
    random.seed(146)
    graph = generate_random_graph(17, 20)
    neighbour_matrix = gen_neighbour_matrix_from_edges(graph.edges, knn=2)
    print(graph.edges)
    print(neighbour_matrix)

