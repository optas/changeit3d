import os
import argparse
import os.path as osp
import numpy as np

####################################
# String handling
####################################

def trim_content_after_last_dot(s):
    """Example: if s = myfile.jpg.png, returns myfile.jpg
    """
    index = s[::-1].find('.') + 1
    s = s[:len(s) - index]
    return s


def create_dir(dir_path):
    """ Creates a directory (or nested directories) if they don't exist.
    """
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path


####################################
# Pointcloud handling
####################################
def rotate_x_axis_by_degrees(pc, theta, clockwise=True):
    """Rotate along x-axis."""
    theta = np.deg2rad(theta)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    rotation_matrix = np.array([[1.0, 0, 0],
                                [0, cos_t, -sin_t],
                                [0, sin_t, cos_t]])
    if not clockwise:
        rotation_matrix = rotation_matrix.T

    return pc.dot(rotation_matrix)


def rotate_y_axis_by_degrees(pc, theta, clockwise=True):
    """Rotate along y-axis."""
    theta = np.deg2rad(theta)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    rotation_matrix = np.array([[cos_t, 0, sin_t],
                                [0, 1.0, 0],
                                [-sin_t, 0, cos_t]])
    if not clockwise:
        rotation_matrix = rotation_matrix.T

    return pc.dot(rotation_matrix)


def rotate_z_axis_by_degrees(pc, theta, clockwise=True):
    """Rotate along z-axis."""
    theta = np.deg2rad(theta)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    rotation_matrix = np.array([[cos_t, -sin_t, 0],
                                [sin_t, cos_t, 0],
                                [0, 0, 1]], dtype=pc.dtype)
    if not clockwise:
        rotation_matrix = rotation_matrix.T

    return pc.dot(rotation_matrix)


def center_in_unit_sphere(pc, in_place=True):
    if not in_place:
        pc = pc.copy()

    for axis in range(3):  # center around each axis
        r_max = np.max(pc[:, axis])
        r_min = np.min(pc[:, axis])
        gap = (r_max + r_min) / 2.0
        pc[:, axis] -= gap

    largest_distance = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc /= largest_distance
    return pc


# def normalize_pointcloud(pc, center_pc=True, norm='sphere', in_place=True):
#     if not in_place:
#         pc = pc.copy()
#
#     if center_pc:
#         for axis in range(3):  # center around each axis
#             r_max = np.max(pc[:, axis])
#             r_min = np.min(pc[:, axis])
#             gap = (r_max + r_min) / 2.0
#             pc[:, axis] -= gap
#
#     if norm == "shpere":
#         largest_distance = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
#     elif norm == "bbox":
#         largest_distance = np.max(np.sum(np.abs(pc), axis=1))
#     else:
#         raise ValueError
#
#     pc /= largest_distance
#     return pc


def swap_axes_of_pointcloud(pointcloud, permutation):
    """
    :param pointcloud: 2-dimensional numpy/torch array: N-points x 3
    :param permutation: a permutation of [0,1,2], e.g., [0,2,1]
    :return:
    """
    v = pointcloud
    nv = len(pointcloud)
    vx = v[:, permutation[0]].reshape(nv, 1)
    vy = v[:, permutation[1]].reshape(nv, 1)
    vz = v[:, permutation[2]].reshape(nv, 1)
    pointcloud = np.hstack((vx, vy, vz))
    return pointcloud

####################################
# Argparse handling
####################################

def str2bool(v):
    """ boolean values for argparse
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def check_rgb_value_type(argvalues):
    """
    Make sure the passed value to argparse is convertible to an RGB value or raise a Type Error.
    Args:
        value: the value to be checked
    """
    if len(argvalues) != 3:
        raise argparse.ArgumentTypeError("%s is an invalid RGB specification" % argvalues)
    for v in argvalues:
        if v < 0 or v > 255:
            raise argparse.ArgumentTypeError("%s is an invalid RGB specification value" % argvalues)
