"""
some helper functions for affine transformations in 2d and 3d
"""

import numpy as np


def translation_matrix(
    tx: float,
    ty: float,
    tz: float,
) -> np.ndarray:
    matrix = [
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1],
    ]

    return np.array(matrix)


def scaling_matrix(
    sx: float,
    sy: float,
    sz: float,
) -> np.ndarray:
    matrix = [
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1],
    ]
    return np.array(matrix)


def rotation_matrix_x(
    alpha: float,
    in_degrees: bool = False,
) -> np.ndarray:
    # pitch
    if in_degrees is True:
        alpha = np.deg2rad(alpha)

    rotation_alpha = [
        [1, 0, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha), 0],
        [0, np.sin(alpha), np.cos(alpha), 0],
        [0, 0, 0, 1],
    ]

    return np.array(rotation_alpha)


def rotation_matrix_y(
    beta: float,
    in_degrees: bool = False,
) -> np.ndarray:
    # roll
    if in_degrees is True:
        beta = np.deg2rad(beta)

    rotation_beta = [
        [np.cos(beta), 0, np.sin(beta), 0],
        [0, 1, 0, 0],
        [-np.sin(beta), 0, np.cos(beta), 0],
        [0, 0, 0, 1],
    ]

    return np.array(rotation_beta)


def rotation_matrix_z(
    gamma: float,
    in_degrees: bool = False,
) -> np.ndarray:
    # yaw
    if in_degrees is True:
        gamma = np.deg2rad(gamma)

    rotation_gamma = [
        [np.cos(gamma), -np.sin(gamma), 0, 0],
        [np.sin(gamma), np.cos(gamma), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]

    return np.array(rotation_gamma)


def rotation_matrix(
    alpha: float,
    beta: float,
    gamma: float,
) -> np.ndarray:
    return rotation_matrix_x(alpha) @ rotation_matrix_y(beta) @ rotation_matrix_z(gamma)


def apply_transform(
    points: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    # points are of shape Nx3, hence transpose
    pt = points.T
    # adding homogeneous coordinate
    ph = np.concatenate([pt, np.ones((1, pt.shape[1]))], axis=0)
    # shaving off the homogeneous coordinate and transpose
    return (transform @ ph)[:-1].T
