"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import os 
os.environ["LEROBOT_HOME"] = "/viscam/projects/dexs2r/data"
import shutil
import h5py 
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from tqdm import tqdm, trange
import zarr
from openpi_client.image_tools import resize_with_pad
from PIL import Image
import torch
from scipy.spatial.transform import Rotation

def rot_mat_to_rot_6d(rot_mat : np.ndarray) -> np.ndarray: 
    """
    Convert a rotation matrix to 6d representation
    rot_mat: N, 3, 3

    return: N, 6
    """
    rot_6d = rot_mat[:, :2, :] # N, 2, 3
    return rot_6d.reshape(-1, 6) # N, 6

def quat_to_rot_6d(quat : np.ndarray, format : str = "wxyz") -> np.ndarray:
    """
    Convert quaternion to 6d representation
    quat: N, 4
    robomimic: 
    https://mujoco.readthedocs.io/en/2.2.1/programming.html#:~:text=To%20represent%203D%20orientations%20and,cos(a%2F2).
    To represent 3D orientations and rotations, MuJoCo uses unit quaternions - namely 4D unit vectors arranged as q = (w, x, y, z). 
    Here (x, y, z) is the rotation axis unit vector scaled by sin(a/2), where a is the rotation angle in radians, and w = cos(a/2). 
    Thus the quaternion corresponding to a null rotation is (1, 0, 0, 0). This is the default setting of all quaternions in MJCF.
    """
    assert format in ["wxyz", "xyzw"], "Invalid quaternion format, only support wxyz or xyzw"
    if format == "wxyz":
        quat = quat[:, [1, 2, 3, 0]]
    rot_mat = Rotation.from_quat(quat).as_matrix()
    return rot_mat_to_rot_6d(rot_mat)

def quat_to_rot_mat(quat : np.ndarray, format : str = "wxyz") -> np.ndarray:
    """
    Convert quaternion to rotation matrix
    quat: N, 4
    """
    assert format in ["wxyz", "xyzw"], "Invalid quaternion format, only support wxyz or xyzw"
    if format == "wxyz":
        quat = quat[:, [1, 2, 3, 0]]
    return Rotation.from_quat(quat).as_matrix()


h5_folder_path = "/juno/u/satvik/dexmachina/gen_data/policy_gen_data_10042_box_486-559_s05_u02_20251031_020846/trajectories"

file_names = os.listdir(h5_folder_path)
file_names = sorted(file_names, key=lambda x: int(x.split("_")[-1][:-3]))[:-1] #the last one is incomplete

resume_demo_idx = 8349
end_demo_idx = 8351

RAW_DATASET_FOLDERS = [
        os.path.join(h5_folder_path, file_name) for file_name in file_names[resume_demo_idx:end_demo_idx]
    ]

LANGUAGE_INSTRUCTIONS = [
    "open the box" for _ in range(len(RAW_DATASET_FOLDERS))
]
REPO_NAME = "dex_open_box_20251031_020846"  # Name of the output dataset, also used for the Hugging Face Hub

CAMERA_KEYS = [
    "trajectory/wrist_cam_left_depth",
    "trajectory/wrist_cam_right_depth"
    
] # folder of rgb images

CAMERA_KEY_MAPPING = {
    "wrist_image_left": "trajectory/wrist_cam_left_depth" ,
    "wrist_image_right": "trajectory/wrist_cam_right_depth",
}

RESIZE_SIZE = 224

def helper_load_human_reference(data):
    # get human wrist poses
    raw_left_wrist_poses = data["trajectory/left_human_wrist_poses"]
    raw_right_wrist_poses = data["trajectory/right_human_wrist_poses"]
    
    left_wrist_poses = np.eye(4)[None,:,:].repeat(len(raw_left_wrist_poses), 0)
    left_wrist_poses[:, :3, 3] = raw_left_wrist_poses[:, :3]
    left_wrist_poses[:, :3, :3] = quat_to_rot_mat(raw_left_wrist_poses[:, 3:])
    
    right_wrist_poses = np.eye(4)[None,:,:].repeat(len(raw_right_wrist_poses), 0)
    right_wrist_poses[:, :3, 3] = raw_right_wrist_poses[:, :3]
    right_wrist_poses[:, :3, :3] = quat_to_rot_mat(raw_right_wrist_poses[:, 3:])
    
    left_write_9d = np.concatenate([
        left_wrist_poses[:, :3, 3],  # positions,
        rot_mat_to_rot_6d(left_wrist_poses[:, :3, :3]) # rotations, 6
    ], axis=1)  # (num_steps, 9)
    right_wrist_9d = np.concatenate([
        right_wrist_poses[:, :3, 3],  # positions,
        rot_mat_to_rot_6d(right_wrist_poses[:, :3, :3]) # rotations, 6
    ], axis=1)  # (num_steps, 9)
    
    human_condition = np.concatenate([left_write_9d, right_wrist_9d], axis=-1)  # (num_pred_steps, 18)
    return human_condition.astype(np.float32).reshape(-1)  # flatten to (18* num_pred_steps, )


def helper_load_proprio(data):
    
    left_wrist_poses = data["trajectory/left_wrist_pose"] #xyz wxyz
    left_joint_poses = data["trajectory/left_joint_qpos"][:]
    right_wrist_poses = data["trajectory/right_wrist_pose"]
    right_joint_poses = data["trajectory/right_joint_qpos"][:]
    left = np.concatenate([
        left_wrist_poses[:,:3],  # positions, 3
        quat_to_rot_6d(left_wrist_poses[:,3:]),  # rotations, 6
        left_joint_poses # left qpos, 27
    ], axis=1)  # (num_steps, 36)
    
    right = np.concatenate([
        right_wrist_poses[:,:3],  # positions, 3
        quat_to_rot_6d(right_wrist_poses[:,3:]),  # rotations, 6
        right_joint_poses # right qpos, 27
    ], axis=1)  # (num_steps, 36)
    proprio = np.concatenate([left, right], axis=-1) 
    
    return proprio.astype(np.float32)


def helper_load_action(data):
    
    # get action path, joint path, and retrieved indices
    left_finger_poses = data["trajectory/left_joint_targets_after_actions"]
    right_finger_poses = data["trajectory/right_joint_targets_after_actions"] #27
    action = np.concatenate([
        left_finger_poses,
        right_finger_poses
    ], axis=-1)  # (num_steps, 54)
    return action.astype(np.float32)

def main():
    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / REPO_NAME
    # if output_path.exists():
    #     shutil.rmtree(output_path)
    print("Dataset saved to ", output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    if output_path.exists():
        print(f"Loading existing dataset from {output_path}")
        dataset = LeRobotDataset(
            repo_id=REPO_NAME,
            local_files_only=True
        )
    else:
        dataset = LeRobotDataset.create(
            repo_id=REPO_NAME,
            robot_type="panda",
            fps=15,
            features={
                # "egocentric_camera": {
                #     "dtype": "video",
                #     "shape": (RESIZE_SIZE, RESIZE_SIZE, 3),
                #     "names": ["height", "width", "channel"],
                # },
                "wrist_image_left": {
                    "dtype": "video",
                    "shape": (RESIZE_SIZE, RESIZE_SIZE, 1),
                    "names": ["height", "width", "channel"],
                },
                "wrist_image_right": {
                    "dtype": "video",
                    "shape": (RESIZE_SIZE, RESIZE_SIZE, 1),
                    "names": ["height", "width", "channel"],
                },
                "human_reference": {
                    "dtype": "float32",
                    "shape": (1314,),
                    "names": ["human_reference"],
                },
                "joint_position": {
                    "dtype": "float32",
                    "shape": (72,),
                    "names": ["joint_position"],
                },
                "actions": {
                    "dtype": "float32",
                    "shape": (54,),
                    "names": ["actions"],
                },
            },
            image_writer_threads=20,
            image_writer_processes=10,
        )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    for raw_dataset_name, language_instruction in zip(RAW_DATASET_FOLDERS, LANGUAGE_INSTRUCTIONS):
        # get all the tasks that are collected that day 
        data_day_dir = raw_dataset_name
        print("Processing file: ", data_day_dir)
        with h5py.File(data_day_dir, "r") as raw_data:
                print(f"Demo is being processed in {data_day_dir}")
                # get the proprio data
                proprio_data = helper_load_proprio(raw_data)
                human_data = helper_load_human_reference(raw_data)
                raw_action = helper_load_action(raw_data)
                seq_length = proprio_data.shape[0]
                last_action = np.zeros_like(raw_action[0])
                images = { key: raw_data[key]for key in CAMERA_KEYS}
                
                for step in tqdm(range(seq_length)):
                    proprio_t = proprio_data[step]
                    action_t = raw_action[step]
                    if np.linalg.norm(action_t - last_action) < 1e-5:
                        continue
                    else:
                        last_action = action_t
                    
                    # get the images for this step
                    images_t = {
                        key: images[CAMERA_KEY_MAPPING[key]][step] for key in CAMERA_KEY_MAPPING
                    }
                    dataset.add_frame(
                        {
                            "joint_position": proprio_t,
                            "actions": action_t,
                            "human_reference": human_data,#human data is constant across time steps
                            **images_t
                        }
                    )
                dataset.save_episode(task=language_instruction)

    # Consolidate the dataset, skip computing stats since we will do that later
    dataset.consolidate(run_compute_stats=False)

    print("Dataset saved to ", output_path)

    # # Optionally push to the Hugging Face Hub
    # dataset.push_to_hub(
    #     tags=["otter", "franka", "pi_0", "multitask"],
    #     private=True,
    #     push_videos=True,
    #     license="apache-2.0",
    # )


if __name__ == "__main__":
    main()
