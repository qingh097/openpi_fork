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


h5_folder_path = "/vision/u/mengdixu/real_data"
file_names = ['r1_pick_cup_real.hdf5']

resume_worker_idx = 0
resume_demo_idx = 45
RAW_DATASET_FOLDERS = [
        os.path.join(h5_folder_path, file_name) for file_name in file_names[resume_worker_idx:]
    ]

LANGUAGE_INSTRUCTIONS = [
    "pick up the cup" for _ in range(len(RAW_DATASET_FOLDERS))
]
REPO_NAME = "r1_real_pick_up_cup_val"  # Name of the output dataset, also used for the Hugging Face Hub


CAMERA_KEYS = [
    "obs/rgb/head/img", 
    "obs/rgb/left_wrist/img",

] # folder of rgb images

CAMERA_KEY_MAPPING = {
    "egocentric_camera": "obs/rgb/head/img",
    "wrist_image_left": "obs/rgb/left_wrist/img",
    "wrist_image_right": "obs/rgb/left_wrist/img",
}

RESIZE_SIZE = 224    

def get_prop_data(raw_data):
    base_qvel = raw_data["obs/odom/base_velocity"][:]
    trunk_qpos = raw_data["obs/joint_state/torso/joint_position"][:]
    arm_left_qpos = raw_data["obs/joint_state/left_arm/joint_position"][:]
    arm_right_qpos = raw_data["obs/joint_state/right_arm/joint_position"][:]
    left_gripper_width = raw_data["obs/gripper_state/left_gripper/gripper_position"][:][:,None]
    right_gripper_width = raw_data["obs/gripper_state/right_gripper/gripper_position"][:][:,None]

    prop_state = np.concatenate((base_qvel, trunk_qpos, arm_left_qpos, arm_right_qpos, left_gripper_width, right_gripper_width), axis=-1) # 21
    return prop_state

def get_actions(raw_data):
     #first 3 base, 4 torso, 7 left arm, 7 right arm
    base = raw_data["action/mobile_base"][:]
    torso = raw_data["action/torso"][:]
    left_arm = raw_data["action/left_arm"][:]
    right_arm = raw_data["action/right_arm"][:]
    left_gripper = raw_data["action/left_gripper"][:][:,None]
    right_gripper = raw_data["action/right_gripper"][:][:,None]
    action = np.concatenate((base, torso, left_arm, left_gripper, right_arm, right_gripper), axis=-1)  # 21
    return action

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
                    "shape": (21,),
                    "names": ["joint_position"],
                },
                "actions": {
                    "dtype": "float32",
                    "shape": (21,),
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
            num_demos = len(raw_data.keys())
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
                demo_data = raw_data[demo_id]
                # demo_data = raw_data[demo_id]
                                
                # get the proprio data
                proprio_data = get_prop_data(demo_data)
                #get action
                raw_action = get_actions(demo_data)
                seq_length = proprio_data.shape[0]
                
                resized_images = {
                    key: resize_with_pad(
                            demo_data[key][...,:3],
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
                    # if np.linalg.norm(action_t - last_action) < 1e-5:
                    #     continue
                    # else:
                    #     last_action = action_t
                    
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
