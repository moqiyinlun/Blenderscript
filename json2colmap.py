from typing import List
import os
import numpy as np
import json
import os
import shutil
def mat2quat(M):
    Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = M.flat
    K = np.array([
        [Qxx - Qyy - Qzz, 0, 0, 0],
        [Qyx + Qxy, Qyy - Qxx - Qzz, 0, 0],
        [Qzx + Qxz, Qzy + Qyz, Qzz - Qxx - Qyy, 0],
        [Qyz - Qzy, Qzx - Qxz, Qxy - Qyx, Qxx + Qyy + Qzz]
    ])
    K /= 3.0
    # Compute eigenvectors (the quaternion components)
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        np.negative(qvec, qvec)
    return qvec

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])
def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def transform_matrix_to_colmap_params(transform_matrix):
    flip_mat = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])

    # Inverse the transformation matrix to switch from world to camera coordinates
    transform_matrix = transform_matrix @ flip_mat
    cam_mat = np.linalg.inv(transform_matrix)
    R = cam_mat[:3, :3]  # Rotation matrix
    t = cam_mat[:3, 3]   # Translation vector
    
    # Convert rotation matrix to quaternion
    qvec = rotmat2qvec(R)
    
    return qvec, t


class Preprocessor:
    def __init__(self, image_path: str,input_json:str, colmap_path: str):
        self.image_path = image_path
        self.input_json = input_json
        self.colmap_path = colmap_path
    def run(self):
        # Step 1: Load input data
        json_file_path = self.input_json
        sparse_dir = os.path.join(self.colmap_path,"sparse/0")
        # Make sure the output directory exists
        os.makedirs(sparse_dir, exist_ok=True)
        # Define output file paths
        cameras_txt_path = os.path.join(sparse_dir, 'cameras.txt')
        images_txt_path = os.path.join(sparse_dir, 'images.txt')
        # Load the JSON data
        with open(json_file_path, 'r') as file:
            transforms_data = json.load(file)
        print(json_file_path)
        # for i in transforms_data['frames']:
        #delete the frame without transform_matrix
            # if i["transform_matrix"] == []:
        # Process the JSON data to generate COLMAP files
        with open(cameras_txt_path, 'w') as f_cam, open(images_txt_path, 'w') as f_img:
            for i, frame in enumerate(transforms_data['frames']):
                # 提取每个帧的宽度、高度和内参矩阵
                width = frame['w']
                height = frame['h']
                # K = frame['K']
                # fx = K[0][0]
                # fy = K[1][1]
                # cx = K[0][2]
                # cy = K[1][2]
                fx = frame['fl_x']
                fy = frame['fl_y']
                cx = frame['cx']
                cy = frame['cy']
                # 处理每个帧的变换矩阵
                transform_mat = np.array(frame['transform_matrix'])
                try:
                    qvec, tvec = transform_matrix_to_colmap_params(transform_mat, flip_mat)
                except:
                    continue
                # 为每个帧写入独立的相机参数
                f_cam.write(f"{i + 1} PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")

                # 写入 images.txt
                image_name = os.path.basename(frame['file_path'])
                qvec_str = ' '.join(map(str, qvec))
                tvec_str = ' '.join(map(str, tvec))
                f_img.write(f"{i + 1} {qvec_str} {tvec_str} {i + 1} {image_name}\n")
                f_img.write("None\n")
res = Preprocessor("","transforms.json","colmap")
res.run()
