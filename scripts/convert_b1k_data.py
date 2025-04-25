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
os.environ["LEROBOT_HOME"] = "/svl/u/ravenh/data"
import shutil
import h5py 
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from tqdm import tqdm, trange
import zarr
from PIL import Image
from openpi_client.image_tools import resize_with_pad

RAW_DATASET_FOLDERS = [
# "/svl/u/mengdixu/b1k-datagen/mimicgen/datasets/demo_450.hdf5"
# "/svl/u/mengdixu/b1k-datagen/mimicgen/datasets/demo_248.hdf5"
"/svl/u/mengdixu/b1k-datagen/brs-algo/datasets/r1_pick_cup_pi.hdf5"
]
LANGUAGE_INSTRUCTIONS = [
    "pick up the green mug"
]
REPO_NAME = "r1_pick_cup_pi"  # Name of the output dataset, also used for the Hugging Face Hub

CAMERA_KEYS = [
    "obs/robot_r1::robot_r1:eyes:Camera:0::rgb", 
    "obs/robot_r1::robot_r1:left_eef_link:Camera:0::rgb",
    "obs/robot_r1::robot_r1:right_eef_link:Camera:0::rgb"
    # "obs/external::viewer::rgb", 
    # "obs/external::viewer::rgb", 
    # "obs/external::viewer::rgb", 
    
] # folder of rgb images
# CAMERA_KEY_MAPPING = {
#     "obs/robot_r1::robot_r1:eyes:Camera:0::rgb": "egocentric_camera",
#     "obs/robot_r1::robot_r1:left_eef_link:Camera:0::rgb": "wrist_image_left",
#     "obs/robot_r1::robot_r1:right_eef_link:Camera:0::rgb": "wrist_image_right",
# }
# CAMERA_KEY_MAPPING = {
#     "egocentric_camera": "obs/external::viewer::rgb",
#     "wrist_image_left": "obs/external::viewer::rgb" ,
#     "wrist_image_right": "obs/external::viewer::rgb",
# }
CAMERA_KEY_MAPPING = {
    "egocentric_camera": "obs/robot_r1::robot_r1:eyes:Camera:0::rgb",
    "wrist_image_left": "obs/robot_r1::robot_r1:left_eef_link:Camera:0::rgb" ,
    "wrist_image_right": "obs/robot_r1::robot_r1:right_eef_link:Camera:0::rgb",
}

STATE_KEY = "obs/prop_state"

RESIZE_SIZE = 224

def main():
    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)
    print("Dataset saved to ", output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
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
    for raw_dataset_name, language_instruction in zip(RAW_DATASET_FOLDERS, LANGUAGE_INSTRUCTIONS):
        # get all the tasks that are collected that day 
        data_day_dir = raw_dataset_name
        print("Processing file: ", data_day_dir)
        raw_data = h5py.File(data_day_dir, "r")
        # get the number of demos
        # num_demos = len(raw_data["data"].keys())
        num_demos = len(raw_data.keys())
        
        for idx in range(num_demos):
        # for idx in range(5):
        
            demo_id = f'demo_{idx}'
            print(f"Demo {idx}/{num_demos}: {demo_id} is being processed")
            # demo_data = raw_data["data"][demo_id]
            demo_data = raw_data[demo_id]
            
            # get the proprio data
            proprio_data = demo_data[STATE_KEY][:]
            #get action
            raw_action = demo_data["actions"][:] #first 3 base, 4 torso, 7 left arm, 7 right arm
            #update action to be delta action but gripper to be absolute
            # action_data = raw_action - proprio_data
            # action_data[:,-8] = raw_action[:,-8] # left gripper
            # action_data[:,-1] = raw_action[:,-1] # right gripper
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
