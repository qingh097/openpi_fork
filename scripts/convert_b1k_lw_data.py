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
# os.environ["LEROBOT_HOME"] = "/svl/u/ravenh/data"
os.environ["LEROBOT_HOME"] = "/vision/u/mengdixu/datasets_pi/ravenh/"
import shutil
import h5py 
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from tqdm import tqdm, trange
import zarr
from PIL import Image
from openpi_client.image_tools import resize_with_pad

h5_folder_path = "/vision/u/wsai/behavior/picking_up_trash/rgbd"

file_names = os.listdir(h5_folder_path)
file_names = sorted(file_names, key=lambda x: int(x.split("_")[-1].split(".")[0])) # sort by the number in the file name

TABLE = False # True for tidy table, False for pick cup
RAW_DATASET_FOLDERS = [
        os.path.join(h5_folder_path, file_name) for file_name in file_names
    ]

LANGUAGE_INSTRUCTIONS = [
    "pick up the trash" for _ in range(len(RAW_DATASET_FOLDERS))
]
REPO_NAME = "b1k_pick_trash_lw"  # Name of the output dataset, also used for the Hugging Face Hub
print("number of demos: ", len(RAW_DATASET_FOLDERS))

CAMERA_KEYS = [
    "obs/robot_r1::robot_r1:zed_link:Camera:0::rgb", 
    "obs/robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
    "obs/robot_r1::robot_r1:right_realsense_link:Camera:0::rgb"
    
] # folder of rgb images

CAMERA_KEY_MAPPING = {
    "egocentric_camera": "obs/robot_r1::robot_r1:zed_link:Camera:0::rgb",
    "wrist_image_left": "obs/robot_r1::robot_r1:left_realsense_link:Camera:0::rgb" ,
    "wrist_image_right": "obs/robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
}

STATE_KEY = "obs/robot_r1::proprio" 

RESIZE_SIZE = 224
resume_demo_idx = 0  # If you want to resume from a specific demo index, set it here

def find_start_point(base_vel):
    """
    Find the first point where the base velocity is non-zero.
    This is used to skip the initial part of the dataset where the robot is not moving.
    """
    start_idx = np.where(np.linalg.norm(base_vel, axis=-1) > 1e-5)[0]
    if len(start_idx) == 0:
        return 0
    return min(start_idx[0],500)  # Limit to the first 100 points to avoid long initial periods

def generate_prop_state(proprio_data):
    base_qvel = proprio_data[:,246:249] # 3
    trunk_qpos = proprio_data[:,238:242] # 4
    arm_left_qpos = proprio_data[:,158:165] #  7
    arm_right_qpos = proprio_data[:,198:205] #  7
    left_gripper_width = proprio_data[:,194:196].sum(axis=-1)[:,None] # 1
    right_gripper_width = proprio_data[:,234:236].sum(axis=-1)[:,None] # 1
    
    prop_state = np.concatenate((base_qvel, trunk_qpos, arm_left_qpos, arm_right_qpos, left_gripper_width, right_gripper_width), axis=-1) # 23
    return prop_state

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
                "egocentric_camera": {
                    "dtype": "video",
                    "shape": (RESIZE_SIZE, RESIZE_SIZE, 3),
                    "names": ["height", "width", "channel"],
                },
                "wrist_image_left": {
                    "dtype": "video",
                    "shape": (RESIZE_SIZE, RESIZE_SIZE, 3),
                    "names": ["height", "width", "channel"],
                },
                "wrist_image_right": {
                    "dtype": "video",
                    "shape": (RESIZE_SIZE, RESIZE_SIZE, 3),
                    "names": ["height", "width", "channel"],
                },
                "joint_position": {
                    "dtype": "float32",
                    "shape": (23,),
                    "names": ["joint_position"],
                },
                "actions": {
                    "dtype": "float32",
                    "shape": (23,),
                    "names": ["actions"],
                },
            },
            image_writer_threads=20,
            image_writer_processes=10,
        )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    first_worker = True
    for raw_dataset_name, language_instruction in zip(RAW_DATASET_FOLDERS, LANGUAGE_INSTRUCTIONS):
        # get all the tasks that are collected that day 
        data_day_dir = raw_dataset_name
        print("Processing file: ", data_day_dir)
        with h5py.File(data_day_dir, "r") as raw_data:
            # raw_data = h5py.File(data_day_dir, "r")
            # get the number of demos
            num_demos = len(raw_data["data"].keys())
            # num_demos = len(raw_data.keys())
            if first_worker:
                start_idx = resume_demo_idx
                first_worker = False
            else:
                start_idx = 0
            for idx in tqdm(range(start_idx, num_demos)):
            # for idx in range(5):
                demo_id = f'demo_{idx}'
                print(f"Demo {idx}/{num_demos}: {demo_id} is being processed in {data_day_dir}")
                demo_data = raw_data["data"][demo_id]
                # get the proprio data
                proprio_data = generate_prop_state(demo_data[STATE_KEY][:]) # 23 joint positions
                #get action
                raw_action = demo_data["action"][:] #first 3 base, 4 torso, 7 left arm, 7 right arm
                #concatenate zero action to the end of the action
                raw_action = np.concatenate([raw_action, np.zeros_like(raw_action[0:1])], axis=0) 
                
                traj_start_idx = find_start_point(raw_action[:, :3]) # 3 base velocities
                proprio_data = proprio_data[traj_start_idx:]
                raw_action = raw_action[traj_start_idx:]
                
                seq_length = proprio_data.shape[0]
                
                resized_images = {
                    key: resize_with_pad(
                            demo_data[key][traj_start_idx:, ...,:3],
                            224,
                            224
                        ) for key in CAMERA_KEYS
                }

                last_action = raw_action[0]
                for step in tqdm(range(seq_length)):
                    # load proprio data
                    proprio_t = proprio_data[step]
                    # create delta action
                    action_t = raw_action[step]
                    if np.linalg.norm(action_t - last_action) < 1e-5:
                        continue
                    else:
                        last_action = action_t
                    
                    # get the images for this step
                    images_t = {
                        key: resized_images[CAMERA_KEY_MAPPING[key]][step] for key in CAMERA_KEY_MAPPING
                    }
                    dataset.add_frame(
                        {
                            "joint_position": proprio_t,
                            "actions": action_t,
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
