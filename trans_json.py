import os
import xml.etree.ElementTree as ET
import math
import numpy as np
# import pymeshlab as ml
from typing import Literal, Tuple , List
import torch
from jaxtyping import Float
from torch import Tensor
import numpy as np
import json

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def rotation_matrix(a: Float[Tensor, "3"], b: Float[Tensor, "3"]) -> Float[Tensor, "3 3"]:
    """Compute the rotation matrix that rotates vector a to vector b.

    Args:
        a: The vector to rotate.
        b: The vector to rotate to.
    Returns:
        The rotation matrix.
    """
    a = a / torch.linalg.norm(a)
    b = b / torch.linalg.norm(b)
    v = torch.cross(a, b)
    c = torch.dot(a, b)
    # If vectors are exactly opposite, we add a little noise to one of them
    if c < -1 + 1e-8:
        eps = (torch.rand(3) - 0.5) * 0.01
        return rotation_matrix(a + eps, b)
    s = torch.linalg.norm(v)
    skew_sym_mat = torch.Tensor(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
    return torch.eye(3) + skew_sym_mat + skew_sym_mat @ skew_sym_mat * ((1 - c) / (s**2 + 1e-8))
def distance(x, os, ds):
    '''Returns the sum distance between point x and all rays'''
    total_distance = 0
    for o, d in zip(os, ds):
        a = o
        b = o + d
        ab = b - a
        ap = x - a
        total_distance += np.linalg.norm(ap - np.dot(ap, ab/np.dot(ab, ab))*ab)
    return total_distance
def auto_orient_and_center_poses(
    poses: Float[Tensor, "*num_poses 4 4"],
    method: Literal["pca", "up", "vertical", "none"] = "up",
    center_method: Literal["poses", "focus", "none"] = "poses",
) -> Tuple[Float[Tensor, "*num_poses 3 4"], Float[Tensor, "3 4"]]:
    """Orients and centers the poses. We provide two methods for orientation: pca and up.

    pca: Orient the poses so that the principal directions of the camera centers are aligned
        with the axes, Z corresponding to the smallest principal component.
        This method works well when all of the cameras are in the same plane, for example when
        images are taken using a mobile robot.
    up: Orient the poses so that the average up vector is aligned with the z axis.
        This method works well when images are not at arbitrary angles.
    vertical: Orient the poses so that the Z 3D direction projects close to the
        y axis in images. This method works better if cameras are not all
        looking in the same 3D direction, which may happen in camera arrays or in LLFF.

    There are two centering methods:
    poses: The poses are centered around the origin.
    focus: The origin is set to the focus of attention of all cameras (the
        closest point to cameras optical axes). Recommended for inward-looking
        camera configurations.

    Args:
        poses: The poses to orient.
        method: The method to use for orientation.
        center_method: The method to use to center the poses.

    Returns:
        Tuple of the oriented poses and the transform matrix.
    """

    origins = poses[..., :3, 3]

    mean_origin = torch.mean(origins, dim=0)
    translation_diff = origins - mean_origin

    if center_method == "poses":
        translation = mean_origin
    elif center_method == "none":
        translation = torch.zeros_like(mean_origin)
    else:
        raise ValueError(f"Unknown value for center_method: {center_method}")

    if method == "pca":
        _, eigvec = torch.linalg.eigh(translation_diff.T @ translation_diff)
        eigvec = torch.flip(eigvec, dims=(-1,))

        if torch.linalg.det(eigvec) < 0:
            eigvec[:, 2] = -eigvec[:, 2]

        transform = torch.cat([eigvec, eigvec @ -translation[..., None]], dim=-1)
        oriented_poses = transform @ poses

        if oriented_poses.mean(dim=0)[2, 1] < 0:
            oriented_poses[:, 1:3] = -1 * oriented_poses[:, 1:3]
    elif method in ("up", "vertical"):
        up = torch.mean(poses[:, :3, 1], dim=0)
        up = up / torch.linalg.norm(up)
        if method == "vertical":
            # If cameras are not all parallel (e.g. not in an LLFF configuration),
            # we can find the 3D direction that most projects vertically in all
            # cameras by minimizing ||Xu|| s.t. ||u||=1. This total least squares
            # problem is solved by SVD.
            x_axis_matrix = poses[:, :3, 0]
            _, S, Vh = torch.linalg.svd(x_axis_matrix, full_matrices=False)
            # Singular values are S_i=||Xv_i|| for each right singular vector v_i.
            # ||S|| = sqrt(n) because lines of X are all unit vectors and the v_i
            # are an orthonormal basis.
            # ||Xv_i|| = sqrt(sum(dot(x_axis_j,v_i)^2)), thus S_i/sqrt(n) is the
            # RMS of cosines between x axes and v_i. If the second smallest singular
            # value corresponds to an angle error less than 10° (cos(80°)=0.17),
            # this is probably a degenerate camera configuration (typical values
            # are around 5° average error for the true vertical). In this case,
            # rather than taking the vector corresponding to the smallest singular
            # value, we project the "up" vector on the plane spanned by the two
            # best singular vectors. We could also just fallback to the "up"
            # solution.
            if S[1] > 0.17 * math.sqrt(poses.shape[0]):
                # regular non-degenerate configuration
                up_vertical = Vh[2, :]
                # It may be pointing up or down. Use "up" to disambiguate the sign.
                up = up_vertical if torch.dot(up_vertical, up) > 0 else -up_vertical
            else:
                # Degenerate configuration: project "up" on the plane spanned by
                # the last two right singular vectors (which are orthogonal to the
                # first). v_0 is a unit vector, no need to divide by its norm when
                # projecting.
                up = up - Vh[0, :] * torch.dot(up, Vh[0, :])
                # re-normalize
                up = up / torch.linalg.norm(up)

        rotation = rotation_matrix(up, torch.Tensor([0, 0, 1]))
        transform = torch.cat([rotation, rotation @ -translation[..., None]], dim=-1)
        oriented_poses = transform @ poses
    elif method == "none":
        transform = torch.eye(4)
        transform[:3, 3] = -translation
        transform = transform[:3, :]
        oriented_poses = transform @ poses
    else:
        raise ValueError(f"Unknown value for method: {method}")

    return oriented_poses, transform
with open("transforms.json") as f:
    calibration_result=json.load(f)
poses=[]
for i in range(len(calibration_result["frames"])):
    pose=np.array(calibration_result["frames"][i]["transform_matrix"])
    poses.append(pose)

poses = torch.from_numpy(np.array(poses).astype(np.float32))

poses, transform_matrix = auto_orient_and_center_poses(
    poses,
    method="up",
    center_method="poses"
)

for i in range(len(calibration_result["frames"])):
    calibration_result["frames"][i]["transform_matrix"]=np.vstack((poses[i], np.array([0, 0, 0, 1]))).tolist()
with open("transforms_new.json", "w") as f:
    json.dump(calibration_result, f, indent=4)