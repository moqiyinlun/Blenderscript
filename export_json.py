import mathutils
import json
import datetime
import bpy
import os
import xml.etree.ElementTree as ET
import numpy as np
from typing import Literal, Tuple , List
def get_scene_mesh_aabb():
    all_world_corners = []

    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue
        local_corners = [mathutils.Vector(corner) for corner in obj.bound_box]
        world_corners = [obj.matrix_world @ corner for corner in local_corners]
        all_world_corners.extend(world_corners)

    if not all_world_corners:
        return None

    coords = np.array([[v.x, v.y, v.z] for v in all_world_corners])
    min_xyz = np.min(coords, axis=0)
    max_xyz = np.max(coords, axis=0)
    return min_xyz.tolist(), max_xyz.tolist()
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def rotation_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if c < -1 + 1e-8:
        eps = (np.random.rand(3) - 0.5) * 0.01
        return rotation_matrix(a + eps, b)
    s = np.linalg.norm(v)
    skew_sym_mat = np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
    return np.eye(3) + skew_sym_mat + skew_sym_mat @ skew_sym_mat * ((1 - c) / (s**2 + 1e-8))

def distance(x, os, ds):
    total_distance = 0
    for o, d in zip(os, ds):
        a = o
        b = o + d
        ab = b - a
        ap = x - a
        total_distance += np.linalg.norm(ap - np.dot(ap, ab/np.dot(ab, ab))*ab)
    return total_distance

def auto_orient_and_center_poses(
    poses: np.ndarray,
    method: Literal["pca", "up", "vertical", "none"] = "up",
    center_method: Literal["poses", "focus", "none"] = "poses",
):
    origins = poses[..., :3, 3]
    mean_origin = np.mean(origins, axis=0)
    translation_diff = origins - mean_origin

    if center_method == "poses":
        translation = mean_origin
    elif center_method == "none":
        translation = np.zeros_like(mean_origin)
    else:
        raise ValueError(f"Unknown value for center_method: {center_method}")

    if method == "pca":
        _, eigvec = np.linalg.eigh(translation_diff.T @ translation_diff)
        eigvec = np.flip(eigvec, axis=-1)

        if np.linalg.det(eigvec) < 0:
            eigvec[:, 2] = -eigvec[:, 2]

        transform = np.concatenate([eigvec, eigvec @ -translation[..., None]], axis=-1)
        oriented_poses = transform @ poses

        if np.mean(oriented_poses, axis=0)[2, 1] < 0:
            oriented_poses[:, 1:3] = -1 * oriented_poses[:, 1:3]

    elif method in ("up", "vertical"):
        up = np.mean(poses[:, :3, 1], axis=0)
        up = up / np.linalg.norm(up)
        if method == "vertical":
            x_axis_matrix = poses[:, :3, 0]
            _, S, Vh = np.linalg.svd(x_axis_matrix, full_matrices=False)
            if S[1] > 0.17 * math.sqrt(poses.shape[0]):
                up_vertical = Vh[2, :]
                up = up_vertical if np.dot(up_vertical, up) > 0 else -up_vertical
            else:
                up = up - Vh[0, :] * np.dot(up, Vh[0, :])
                up = up / np.linalg.norm(up)

        rotation = rotation_matrix(up, np.array([0, 0, 1]))
        transform = np.concatenate([rotation, rotation @ -translation[..., None]], axis=-1)
        oriented_poses = transform @ poses

    elif method == "none":
        transform = np.eye(4)
        transform[:3, 3] = -translation
        transform = transform[:3, :]
        oriented_poses = transform @ poses
    else:
        raise ValueError(f"Unknown value for method: {method}")

    return oriented_poses, transform

# global addon script variables
OUTPUT_TRAIN = 'train'
OUTPUT_TEST = 'test'
CAMERA_NAME = 'BlenderNeRF Camera'


# blender nerf operator parent class
class BlenderNeRF_Operator():

    # camera intrinsics
    def get_camera_intrinsics(self, scene, camera):
        camera_angle_x = camera.data.angle_x
        camera_angle_y = camera.data.angle_y

        # camera properties
        f_in_mm = camera.data.lens # focal length in mm
        scale = scene.render.resolution_percentage / 100
        width_res_in_px = scene.render.resolution_x * scale # width
        height_res_in_px = scene.render.resolution_y * scale # height
        optical_center_x = width_res_in_px / 2
        optical_center_y = height_res_in_px / 2

        # pixel aspect ratios
        size_x = scene.render.pixel_aspect_x * width_res_in_px
        size_y = scene.render.pixel_aspect_y * height_res_in_px
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

        # sensor fit and sensor size (and camera angle swap in specific cases)
        if camera.data.sensor_fit == 'AUTO':
            sensor_size_in_mm = camera.data.sensor_height if width_res_in_px < height_res_in_px else camera.data.sensor_width
            if width_res_in_px < height_res_in_px:
                sensor_fit = 'VERTICAL'
                camera_angle_x, camera_angle_y = camera_angle_y, camera_angle_x
            elif width_res_in_px > height_res_in_px:
                sensor_fit = 'HORIZONTAL'
            else:
                sensor_fit = 'VERTICAL' if size_x <= size_y else 'HORIZONTAL'

        else:
            sensor_fit = camera.data.sensor_fit
            if sensor_fit == 'VERTICAL':
                sensor_size_in_mm = camera.data.sensor_height if width_res_in_px <= height_res_in_px else camera.data.sensor_width
                if width_res_in_px <= height_res_in_px:
                    camera_angle_x, camera_angle_y = camera_angle_y, camera_angle_x

        # focal length for horizontal sensor fit
        if sensor_fit == 'HORIZONTAL':
            sensor_size_in_mm = camera.data.sensor_width
            s_u = f_in_mm / sensor_size_in_mm * width_res_in_px
            s_v = f_in_mm / sensor_size_in_mm * width_res_in_px * pixel_aspect_ratio

        # focal length for vertical sensor fit
        if sensor_fit == 'VERTICAL':
            s_u = f_in_mm / sensor_size_in_mm * width_res_in_px / pixel_aspect_ratio
            s_v = f_in_mm / sensor_size_in_mm * width_res_in_px

        camera_intr_dict = {
            'camera_angle_x': camera_angle_x,
            'camera_angle_y': camera_angle_y,
            'fl_x': s_u,
            'fl_y': s_v,
            'k1': 0.0,
            'k2': 0.0,
            'p1': 0.0,
            'p2': 0.0,
            'cx': optical_center_x,
            'cy': optical_center_y,
            'w': width_res_in_px,
            'h': height_res_in_px,
        }

        return camera_intr_dict

    # camera extrinsics (transform matrices)
    def get_camera_extrinsics(self, scene, camera, mode='TRAIN', method='SOF'):
        assert mode == 'TRAIN' or mode == 'TEST'
        assert method == 'SOF' or method == 'TTC' or method == 'COS'

        initFrame = scene.frame_current
        step = 1
        if (mode == 'TRAIN' and method == 'COS'):
            end = scene.frame_start + scene.cos_nb_frames - 1
        elif (mode == 'TRAIN' and method == 'TTC'):
            end = scene.frame_start + scene.ttc_nb_frames - 1
        else:
            end = scene.frame_end

        camera_extr_dict = []
        for frame in range(1, 2, step):
            scene.frame_set(frame)
            filename = os.path.basename( scene.render.frame_path(frame=frame) )
            filedir = OUTPUT_TRAIN * (mode == 'TRAIN') + OUTPUT_TEST * (mode == 'TEST')

            frame_data = {
                'file_path': os.path.join(filedir, filename),
                'transform_matrix': self.listify_matrix( camera.matrix_world )
            }

            camera_extr_dict.append(frame_data)

        scene.frame_set(initFrame) # set back to initial frame

        return camera_extr_dict

    def save_json(self, directory, filename, data, indent=4):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'w') as file:
            json.dump(data, file, indent=indent)

    def is_power_of_two(self, x):
        return math.log2(x).is_integer()

    # function from original nerf 360_view.py code for blender
    def listify_matrix(self, matrix):
        matrix_list = []
        for row in matrix:
            matrix_list.append(list(row))
        return matrix_list

    # assert messages
    def asserts(self, scene, method='SOF'):
        assert method == 'SOF' or method == 'TTC' or method == 'COS'

        camera = scene.camera
        train_camera = scene.camera_train_target
        test_camera = scene.camera_test_target

        sof_name = scene.sof_dataset_name
        ttc_name = scene.ttc_dataset_name
        cos_name = scene.cos_dataset_name

        error_messages = []

        if (method == 'SOF' or method == 'COS') and not camera.data.type == 'PERSP':
            error_messages.append('Only perspective cameras are supported!')

        if method == 'TTC' and not (train_camera.data.type == 'PERSP' and test_camera.data.type == 'PERSP'):
           error_messages.append('Only perspective cameras are supported!')

        if method == 'COS' and CAMERA_NAME in scene.objects.keys():
            sphere_camera = scene.objects[CAMERA_NAME]
            if not sphere_camera.data.type == 'PERSP':
                error_messages.append('BlenderNeRF Camera must remain a perspective camera!')

        if (method == 'SOF' and sof_name == '') or (method == 'TTC' and ttc_name == '') or (method == 'COS' and cos_name == ''):
            error_messages.append('Dataset name cannot be empty!')

        if method == 'COS' and any(x == 0 for x in scene.sphere_scale):
            error_messages.append('The BlenderNeRF Sphere cannot be flat! Change its scale to be non zero in all axes.')

        if not scene.nerf and not self.is_power_of_two(scene.aabb):
            error_messages.append('AABB scale needs to be a power of two!')

        if scene.save_path == '':
            error_messages.append('Save path cannot be empty!')

        return error_messages

    def save_log_file(self, scene, directory, method='SOF'):
        assert method == 'SOF' or method == 'TTC' or method == 'COS'
        now = datetime.datetime.now()

        logdata = {
            'BlenderNeRF Version': scene.blendernerf_version,
            'Date and Time' : now.strftime("%d/%m/%Y %H:%M:%S"),
            'Train': scene.train_data,
            'Test': scene.test_data,
            'AABB': scene.aabb,
            'Render Frames': scene.render_frames,
            'File Format': 'NeRF' if scene.nerf else 'NGP',
            'Save Path': scene.save_path,
            'Method': method
        }

        if method == 'SOF':
            logdata['Frame Step'] = scene.train_frame_steps
            logdata['Camera'] = scene.camera.name
            logdata['Dataset Name'] = scene.sof_dataset_name

        elif method == 'TTC':
            logdata['Train Camera Name'] = scene.camera_train_target.name
            logdata['Test Camera Name'] = scene.camera_test_target.name
            logdata['Frames'] = scene.ttc_nb_frames
            logdata['Dataset Name'] = scene.ttc_dataset_name

        else:
            logdata['Camera'] = scene.camera.name
            logdata['Location'] = str(list(scene.sphere_location))
            logdata['Rotation'] = str(list(scene.sphere_rotation))
            logdata['Scale'] = str(list(scene.sphere_scale))
            logdata['Radius'] = scene.sphere_radius
            logdata['Lens'] = str(scene.focal) + ' mm'
            logdata['Seed'] = scene.seed
            logdata['Frames'] = scene.cos_nb_frames
            logdata['Upper Views'] = scene.upper_views
            logdata['Outwards'] = scene.outwards
            logdata['Dataset Name'] = scene.cos_dataset_name

        self.save_json(directory, filename='log.txt', data=logdata)
    def export_all_cameras(self, context):
        scene = context.scene
        cameras_in_scene = [obj for obj in scene.objects if obj.type == 'CAMERA'] 

        all_camera_data = {}
        all_camera_data["frames"] = []
        pose_data = []
        for cam in cameras_in_scene:
            extrinsics = self.get_camera_extrinsics(scene, cam)
            pose_data.append(np.array(extrinsics[0]["transform_matrix"]))
        # pose_data = torch.from_numpy(np.array(pose_data).astype(np.float32))
        pose_data = np.array(pose_data)
#        pose_data, transform_matrix = auto_orient_and_center_poses(
#            pose_data,
#            method="up",
#            center_method="poses"
#        )
        cnt = 0 
        for cam in cameras_in_scene:
            temp_frame = {}
            intrinsics = self.get_camera_intrinsics(scene, cam)
            extrinsics = self.get_camera_extrinsics(scene, cam) 
            for key in intrinsics.keys():
                temp_frame[key] = intrinsics[key]
            temp_frame["transform_matrix"] = extrinsics[0]["transform_matrix"]
            temp_frame["file_path"] = cam.name+".png"
            all_camera_data["frames"].append(temp_frame)
            cnt+=1
        aabb = get_scene_mesh_aabb()
        print(aabb)
        aabb = np.array(aabb)
        center = (aabb[0] + aabb[1]) / 2
        half_size = (aabb[1] - aabb[0]) / 2

        # Scale
        scaled_half_size = half_size * 1.5
        scaled_aabb = [(center - scaled_half_size).tolist(), (center + scaled_half_size).tolist()]
        all_camera_data["aabb"] = scaled_aabb #[[aabb[0][1],aabb[0][2],aabb[0][0]],[aabb[1][1],aabb[1][2],aabb[1][0]]]
        output_directory = r"C:\moqiyinlun\3DPrinter\Hair"  
        self.save_json(output_directory, "transforms.json", all_camera_data)

context = bpy.context

operator = BlenderNeRF_Operator()
operator.export_all_cameras(context)