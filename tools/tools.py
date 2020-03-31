
from __future__ import division

import numpy as np
import os, sys
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.switch_backend('agg')
from  matplotlib import ticker
from matplotlib import colors
import itertools

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')

def project_point_radial( P, R, T, f, c, k, p ):
    """
    Project points from 3d to 2d using camera parameters
    including radial and tangential distortion

    Args
        P: Nx3 points in world coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
        f: (scalar) Camera focal length
        c: 2x1 Camera center
        k: 3x1 Camera radial distortion coefficients
        p: 2x1 Camera tangential distortion coefficients
    Returns
        Proj: Nx2 points in pixel space
        D: 1xN depth of each point in camera space
        radial: 1xN radial distortion per point
        tan: 1xN tangential distortion per point
        r2: 1xN squared radius of the projected points before distortion
    """

    # P is a matrix of 3-dimensional points
    assert len(P.shape) == 2
    assert P.shape[1] == 3

    N = P.shape[0]
    X = R.dot( P.T - T ) # rotate and translate
    XX = X[:2,:] / X[2,:]
    r2 = XX[0,:]**2 + XX[1,:]**2

    radial = 1 + np.einsum( 'ij,ij->j', np.tile(k,(1, N)), np.array([r2, r2**2, r2**3]) );
    tan = p[0]*XX[1,:] + p[1]*XX[0,:]

    XXX = XX * np.tile(radial+tan,(2,1)) + np.outer(np.array([p[1], p[0]]).reshape(-1), r2 )

    Proj = (f * XXX) + c
    Proj = Proj.T

    D = X[2,]

    return Proj, D, radial, tan, r2


def world_to_camera_frame(P, R, T):
    """
    Convert points from world to camera coordinates

    Args
        P: Nx3 3d points in world coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
    Returns
        X_cam: Nx3 3d points in camera coordinates
    """

    assert len(P.shape) == 2
    assert P.shape[1] == 3

    X_cam = R.dot( P.T - T ) # rotate and translate

    return X_cam.T

def camera_to_world_frame(P, R, T):
    """Inverse of world_to_camera_frame

    Args
        P: Nx3 points in camera coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
    Returns
        X_cam: Nx3 points in world coordinates
    """

    assert len(P.shape) == 2
    assert P.shape[1] == 3

    X_cam = R.T.dot( P.T ) + T # rotate and translate

    return X_cam.T

def procrustes(A, B, scaling=True, reflection='best'):
    """ A port of MATLAB's `procrustes` function to Numpy.

    $$ \min_{R, T, S} \sum_i^N || A_i - R B_i + T ||^2. $$
    Use notation from [course note]
    (https://fling.seas.upenn.edu/~cis390/dynamic/slides/CIS390_Lecture11.pdf).

    Args:
        A: Matrices of target coordinates.
        B: Matrices of input coordinates. Must have equal numbers of  points
            (rows), but B may have fewer dimensions (columns) than A.
        scaling: if False, the scaling component of the transformation is forced
            to 1
        reflection:
            if 'best' (default), the transformation solution may or may not
            include a reflection component, depending on which fits the data
            best. setting reflection to True or False forces a solution with
            reflection or no reflection respectively.

    Returns:
        d: The residual sum of squared errors, normalized according to a measure
            of the scale of A, ((A - A.mean(0))**2).sum().
        Z: The matrix of transformed B-values.
        tform: A dict specifying the rotation, translation and scaling that
            maps A --> B.
    """
    assert A.shape[0] == B.shape[0]
    n, dim_x = A.shape
    _, dim_y = B.shape

    # remove translation
    A_bar = A.mean(0)
    B_bar = B.mean(0)
    A0 = A - A_bar
    B0 = B - B_bar

    # remove scale
    ssX = (A0**2).sum()
    ssY = (B0**2).sum()
    A_norm = np.sqrt(ssX)
    B_norm = np.sqrt(ssY)
    A0 /= A_norm
    B0 /= B_norm

    if dim_y < dim_x:
        B0 = np.concatenate((B0, np.zeros(n, dim_x - dim_y)), 0)

    # optimum rotation matrix of B
    A = np.dot(A0.T, B0)
    U, s, Vt = np.linalg.svd(A)
    V = Vt.T
    R = np.dot(V, U.T)

    if reflection is not 'best':
        # does the current solution use a reflection?
        have_reflection = np.linalg.det(R) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            R = np.dot(V, U.T)

    S_trace = s.sum()
    if scaling:
        # optimum scaling of B
        scale = S_trace * A_norm / B_norm

        # standarised distance between A and scale*B*R + c
        d = 1 - S_trace**2

        # transformed coords
        Z = A_norm * S_trace * np.dot(B0, R) + A_bar
    else:
        scale = 1
        d = 1 + ssY / ssX - 2 * S_trace * B_norm / A_norm
        Z = B_norm * np.dot(B0, R) + A_bar

    # transformation matrix
    if dim_y < dim_x:
        R = R[:dim_y, :]
    translation = A_bar - scale * np.dot(B_bar, R)

    # transformation values
    tform = {'rotation': R, 'scale': scale, 'translation': translation}
    return d, Z, tform

def image_to_camera_frame(pose3d_image_frame, box, camera, rootIdx, root_depth):
    rectangle_3d_size = 2000.0
    ratio = (box[2] - box[0] + 1) / rectangle_3d_size
    pose3d_image_frame = pose3d_image_frame.copy()
    pose3d_image_frame[:, 2] = pose3d_image_frame[:, 2] / ratio + root_depth

    cx, cy, fx, fy = camera['cx'], camera['cy'], camera['fx'], camera['fy']
    pose3d_image_frame[:, 0] = (pose3d_image_frame[:, 0] - cx) / fx
    pose3d_image_frame[:, 1] = (pose3d_image_frame[:, 1] - cy) / fy
    pose3d_image_frame[:, 0] *= pose3d_image_frame[:, 2]
    pose3d_image_frame[:, 1] *= pose3d_image_frame[:, 2]
    return pose3d_image_frame


def align_to_gt(pose, pose_gt):
    """Align pose to ground truth pose.

    Use MLE.
    """
    return procrustes(pose_gt, pose)[1]

