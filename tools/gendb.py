# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

import argparse
import os
import os.path as osp
import scipy.io as sio
import numpy as np
import parse
import cameras
import pickle


def find_train_val_dirs(dataset_root_dir):
    trainsubs = ['s_01', 's_05', 's_06', 's_07', 's_08']
    train_dirs, val_dirs = [], []
    allsets = os.listdir(dataset_root_dir)
    for dset in allsets:
        if osp.isdir(osp.join(dataset_root_dir, dset)):
            if dset[:4] in trainsubs:
                train_dirs.append(dset)
            else:
                val_dirs.append(dset)
    train_dirs.sort()
    val_dirs.sort()
    return train_dirs, val_dirs


def infer_meta_from_name(datadir):
    format_str = 's_{}_act_{}_subact_{}_ca_{}'
    res = parse.parse(format_str, datadir)
    meta = {
        'subject': int(res[0]),
        'action': int(res[1]),
        'subaction': int(res[2]),
        'camera': int(res[3]) - 1
    }
    return meta


def load_db(dataset_root_dir, dset, vid, cams, rootIdx=0):
    annofile = os.path.join(dataset_root_dir, dset, 'matlab_meta.mat')
    anno = sio.loadmat(annofile)
    numimgs = anno['num_images'][0][0]
    joints_3d_cam = np.reshape(np.transpose(anno['Y3d_mono']), (numimgs, -1, 3))
    meta = infer_meta_from_name(dset)
    cam = _retrieve_camera(cams, meta['subject'], meta['camera'])

    dataset = []
    for i in range(numimgs):
        image = 's_{:02}_act_{:02}_subact_{:02}_ca_{:02}_{:06}.jpg'.format(
            meta['subject'], meta['action'], meta['subaction'],
            meta['camera'] + 1, i + 1)
        image = os.path.join(dset, image)
        joint_3d_cam = joints_3d_cam[i]
        box = _infer_box(joint_3d_cam, cam, rootIdx)
        joint_3d_image = camera_to_image_frame(joint_3d_cam, box, cam, rootIdx)
        center = (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))
        scale = ((box[2] - box[0]) / 200.0, (box[3] - box[1]) / 200.0)
        dataitem = {
            'videoid': vid,
            'cameraid': meta['camera'],
            'camera_param': cam,
            'imageid': i,
            'image_path': image,
            'joint_3d_image': joint_3d_image,
            'joint_3d_camera': joint_3d_cam,
            'center': center,
            'scale': scale,
            'box': box,
            'subject': meta['subject'],
            'action': meta['action'],
            'subaction': meta['subaction'],
            'root_depth': joint_3d_cam[rootIdx, 2]
        }

        dataset.append(dataitem)
    return dataset


def camera_to_image_frame(pose3d, box, camera, rootIdx):
    rectangle_3d_size = 2000.0
    ratio = (box[2] - box[0] + 1) / rectangle_3d_size
    pose3d_image_frame = np.zeros_like(pose3d)
    pose3d_image_frame[:, :2] = _weak_project(
        pose3d.copy(), camera['fx'], camera['fy'], camera['cx'], camera['cy'])
    pose3d_depth = ratio * (pose3d[:, 2] - pose3d[rootIdx, 2])
    pose3d_image_frame[:, 2] = pose3d_depth
    return pose3d_image_frame


def image_to_camera_frame(pose3d_image_frame, box, camera, rootIdx, root_depth):
    rectangle_3d_size = 2000.0
    ratio = (box[2] - box[0] + 1) / rectangle_3d_size
    pose3d_image_frame[:, 2] = pose3d_image_frame[:, 2] / ratio + root_depth

    cx, cy, fx, fy = camera['cx'], camera['cy'], camera['fx'], camera['fy']
    pose3d_image_frame[:, 0] = (pose3d_image_frame[:, 0] - cx) / fx
    pose3d_image_frame[:, 1] = (pose3d_image_frame[:, 1] - cy) / fy
    pose3d_image_frame[:, 0] *= pose3d_image_frame[:, 2]
    pose3d_image_frame[:, 1] *= pose3d_image_frame[:, 2]
    return pose3d_image_frame


def _infer_box(pose3d, camera, rootIdx):
    root_joint = pose3d[rootIdx, :]
    tl_joint = root_joint.copy()
    tl_joint[:2] -= 1000.0
    br_joint = root_joint.copy()
    br_joint[:2] += 1000.0
    tl_joint = np.reshape(tl_joint, (1, 3))
    br_joint = np.reshape(br_joint, (1, 3))

    tl2d = _weak_project(tl_joint, camera['fx'], camera['fy'], camera['cx'],
                         camera['cy']).flatten()

    br2d = _weak_project(br_joint, camera['fx'], camera['fy'], camera['cx'],
                         camera['cy']).flatten()
    return np.array([tl2d[0], tl2d[1], br2d[0], br2d[1]])


def _weak_project(pose3d, fx, fy, cx, cy):
    pose2d = pose3d[:, :2] / pose3d[:, 2:3]
    pose2d[:, 0] *= fx
    pose2d[:, 1] *= fy
    pose2d[:, 0] += cx
    pose2d[:, 1] += cy
    return pose2d


def _retrieve_camera(cameras, subject, cameraidx):
    R, T, f, c, k, p, name = cameras[(subject, cameraidx + 1)]
    camera = {}
    camera['R'] = R
    camera['T'] = T
    camera['fx'] = f[0]
    camera['fy'] = f[1]
    camera['cx'] = c[0]
    camera['cy'] = c[1]
    camera['k'] = k
    camera['p'] = p
    camera['name'] = name
    return camera


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_root_dir')
    args = parser.parse_args()

    cams = cameras.load_cameras(
        bpath=os.path.join(args.dataset_root_dir, 'cameras.h5'))

    train_dirs, val_dirs = find_train_val_dirs(args.dataset_root_dir)
    train_val_datasets = [train_dirs, val_dirs]
    dbs = []
    video_count = 0
    for dataset in train_val_datasets:
        db = []
        for video in dataset:
            if np.mod(video_count, 1) == 0:
                print('Process {}: {}'.format(video_count, video))

            data = load_db(args.dataset_root_dir, video, video_count, cams)
            db.extend(data)
            video_count += 1
        dbs.append(db)

    datasets = {'train': dbs[0], 'validation': dbs[1]}

    with open(args.dataset_root_dir + 'h36m_train.pkl', 'wb') as f:
        pickle.dump(datasets['train'], f)

    with open(args.dataset_root_dir + 'h36m_test.pkl', 'wb') as f:
        pickle.dump(datasets['validation'], f)
