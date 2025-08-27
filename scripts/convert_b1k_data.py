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

# h5_folder_path = "/vision/u/chengshu/momagen/pick_cup_full"
# h5_folder_path = '/vision/u/chengshu/momagen/pick_cup_no_vis'
# h5_folder_path = '/vision/u/chengshu/momagen/pick_cup_only_hard'
# h5_folder_path = '/vision/u/chengshu/momagen/tidy_table_full'
h5_folder_path = "/vision/u/chengshu/mimicgen/pick_cup_full"
h5_folder_path = "/vision/u/chengshu/skillgen/pick_cup_full"
h5_folder_path = "/vision/u/chengshu/momagen/tidy_table_full"
h5_folder_path = "/vision/u/chengshu/mimicgen/tidy_table_full"
h5_folder_path = "/vision/u/chengshu/skillgen/tidy_table_full"
h5_folder_path = "/vision/u/chengshu/momagen/tidy_table_no_vis"
# h5_folder_path = "/vision/u/chengshu/momagen/tidy_table_only_hard"
h5_folder_path = "/vision/u/chengshu/momagen/tidy_table_only_soft"
h5_folder_path = "/vision/u/chengshu/momagen/clean_pan_full"

file_names = os.listdir(h5_folder_path)
file_names = sorted(file_names, key=lambda x: int(x.split("_")[-1]))

resume_worker_idx = 3
resume_demo_idx = 28

TABLE = False # True for tidy table, False for pick cup
# RAW_DATASET_FOLDERS = [
#         os.path.join(h5_folder_path, file_name, "demo_src_r1_pick_cup_task_D1", "demo.hdf5") for file_name in file_names
#     ]
# RAW_DATASET_FOLDERS = [
#         os.path.join(h5_folder_path, file_name, "demo_src_r1_pick_cup_mimicgen_task_D0", "demo.hdf5") for file_name in file_names
#     ]
# RAW_DATASET_FOLDERS = [
#         os.path.join(h5_folder_path, file_name, "demo_src_r1_pick_cup_skillgen_task_D0", "demo.hdf5") for file_name in file_names
#     ]

# LANGUAGE_INSTRUCTIONS = [
#     "pick up the green mug" for _ in range(len(RAW_DATASET_FOLDERS))
# ]
# REPO_NAME = "r1_pick_cup_skillgen_D0"  # Name of the output dataset, also used for the Hugging Face Hub

# TABLE = True # True for tidy table, False for pick cup

# RAW_DATASET_FOLDERS = [
#         os.path.join(h5_folder_path, file_name, "demo_src_r1_tidy_table_task_D0", "demo.hdf5") for file_name in file_names[resume_worker_idx:]
#     ]

# LANGUAGE_INSTRUCTIONS = [
#     "pick up the mug and place in the sink" for _ in range(len(RAW_DATASET_FOLDERS))
# ]
# REPO_NAME = "r1_tidy_table_only_soft_D0"  # Name of the output dataset, also used for the Hugging Face Hub

PAN = True # True for tidy table, False for pick cup

RAW_DATASET_FOLDERS = [
        os.path.join(h5_folder_path, file_name, "demo_src_r1_clean_pan_task_D0", "demo.hdf5") for file_name in file_names[resume_worker_idx:]
    ]

LANGUAGE_INSTRUCTIONS = [
    "clean the pan" for _ in range(len(RAW_DATASET_FOLDERS))
]
REPO_NAME = "r1_clean_pan_D0"  # Name of the output dataset, also used for the Hugging Face Hub


CAMERA_KEYS = [
    "obs/robot_r1::robot_r1:eyes:Camera:0::rgb", 
    "obs/robot_r1::robot_r1:left_eef_link:Camera:0::rgb",
    "obs/robot_r1::robot_r1:right_eef_link:Camera:0::rgb"
    
] # folder of rgb images

CAMERA_KEY_MAPPING = {
    "egocentric_camera": "obs/robot_r1::robot_r1:eyes:Camera:0::rgb",
    "wrist_image_left": "obs/robot_r1::robot_r1:left_eef_link:Camera:0::rgb" ,
    "wrist_image_right": "obs/robot_r1::robot_r1:right_eef_link:Camera:0::rgb",
}

STATE_KEY = "obs/prop_state"

RESIZE_SIZE = 224


def clip_traj_based_on_cup_pose(cup_pose):
    act_mask = np.ones(cup_pose.shape[0], dtype=bool)
    cup_z = cup_pose[:, 2]
    act_mask[cup_z < 0.9] = True
    print('clip based on cup')
    print('remaining steps', sum(act_mask))
    print('original length', len(act_mask))
    return act_mask

def remove_no_act_segment(traj, gripper_position, seg_mask, window_size=5, tol=1e-3):
    # construct a mask for the trajectory
    act_mask = np.ones(traj.shape[0], dtype=bool)
    # get the index where seg_mask is 1
    start_idx = np.where(seg_mask)[0][0]
    # staring from start_idx
    for i in range(start_idx, len(seg_mask)-window_size):
        # get the following window_size 
        start_pose = np.concatenate([traj[i], gripper_position[i][None]])
        end_pose = np.concatenate([traj[i+window_size], gripper_position[i+window_size][None]])
        if np.allclose(start_pose, end_pose, atol=tol):
            act_mask[i+1:i+window_size+1] = False
    return act_mask


def remove_no_act_segment_tidy_table(traj, gripper_position, seg_mask, window_size=5, tol=1e-3):
    # construct a mask for the trajectory
    act_mask = np.ones(traj.shape[0], dtype=bool)
    for i in range(0, len(seg_mask)-window_size):
        # get the following window_size 
        start_pose = np.concatenate([traj[i], gripper_position[i][None]])
        end_pose = np.concatenate([traj[i+window_size], gripper_position[i+window_size][None]])
        if np.allclose(start_pose, end_pose, atol=tol):
            for i_i in range(i+1, i+window_size+1):
                if seg_mask[i_i] == 1:
                    # only remove the step if it is in the replay segment
                    act_mask[i_i] = False
                    # print('remove segment', i_i)

    # print('act mask:', act_mask)
    # print('remove no act segment')
    # print('reamaining steps', sum(act_mask))
    # print('original length', len(act_mask))
    # breakpoint()
    return act_mask


def remove_no_act_segment_clean_pan(traj, gripper_position, seg_mask, window_size=5, tol=1e-3):
    # construct a mask for the trajectory
    act_mask = np.ones(traj.shape[0], dtype=bool)
    for i in range(0, len(seg_mask)-window_size):
        # get the following window_size 
        start_pose = np.concatenate([traj[i], gripper_position[i][None]])
        end_pose = np.concatenate([traj[i+window_size], gripper_position[i+window_size][None]])
        if np.allclose(start_pose, end_pose, atol=tol):
            for i_i in range(i+1, i+window_size+1):
                if seg_mask[i_i] == 1:
                    # only remove the step if it is in the replay segment
                    act_mask[i_i] = False
                    # print('remove segment', i_i)
    return act_mask

def process_seg_mask_clean_pan(demo_data):
    # customize for the clean pan task 
    # mobile mp, arm mp, arm replay, mobile mp, arm mp, arm replay
    # mobile mp, arm_mp, arm replay, mobile mp, arm mp, arm replay, arm_mp, arm replay
    # process segmentation mask 
    num_steps = demo_data['actions'].shape[0] - 1
    subtask_lengths = np.array(demo_data["subtask_lengths"])
    left_mp_ranges = np.array(demo_data["left_mp_ranges"])
    right_mp_ranges = np.array(demo_data["right_mp_ranges"])

    # for replay mask, 0, 0, 1, 0, 0, 1
    # for replay mask, 0, 0, 1, 0, 0, 1, 0, 1
    replay_seg_mask = np.zeros((num_steps, 1), dtype=bool)
    replay_seg_mask[left_mp_ranges[1,1]:left_mp_ranges[2,0], 0] = True
    replay_seg_mask[left_mp_ranges[3,1]:left_mp_ranges[4,0], 0] = True
    replay_seg_mask[left_mp_ranges[4,1]:, 0] = True

    # for manipulation mask, 0, 1, 1, 0, 1, 1
    # for manipulation mask, 0, 1, 1, 0, 1, 1, 1, 1
    manip_seg_mask = np.zeros((num_steps, 1), dtype=bool)
    manip_seg_mask[left_mp_ranges[1,0]:left_mp_ranges[2,0], 0] = True
    manip_seg_mask[left_mp_ranges[3,0]:, 0] = True

    return replay_seg_mask, manip_seg_mask

def process_seg_mask_nav_manip(demo_data):
    # process segmentation mask 
    num_steps = demo_data['actions'].shape[0] - 1
    subtask_lengths = np.array(demo_data["subtask_lengths"])
    left_mp_ranges = np.array(demo_data["left_mp_ranges"])
    right_mp_ranges = np.array(demo_data["right_mp_ranges"])
    # TODO!!! hacky for r1_pick, 0 for navigation, 0 for mp in manipulation, 1 for replay part
    seg_mask = np.zeros((num_steps, 1), dtype=bool)
    seg_mask[left_mp_ranges[1,0]:left_mp_ranges[1,1], 0] = False
    seg_mask[left_mp_ranges[1,1]:, 0] = True
    return seg_mask



def process_seg_mask_nav_manip_tidy_table(demo_data):
    # customize for the tidy table task 
    # mobile mp, arm mp, arm replay, mobile mp, arm mp, arm replay
    # process segmentation mask 
    num_steps = demo_data['actions'].shape[0] - 1
    subtask_lengths = np.array(demo_data["subtask_lengths"])
    left_mp_ranges = np.array(demo_data["left_mp_ranges"])
    right_mp_ranges = np.array(demo_data["right_mp_ranges"])

    # for replay mask, 0, 0, 1, 0, 0, 1
    replay_seg_mask = np.zeros((num_steps, 1), dtype=bool)
    replay_seg_mask[left_mp_ranges[1,1]:left_mp_ranges[2,0], 0] = True
    replay_seg_mask[left_mp_ranges[3,1]:, 0] = True

    # for manipulation mask, 0, 1, 1, 0, 1, 1
    manip_seg_mask = np.zeros((num_steps, 1), dtype=bool)
    manip_seg_mask[left_mp_ranges[1,0]:left_mp_ranges[2,0], 0] = True
    manip_seg_mask[left_mp_ranges[3,0]:, 0] = True

    return replay_seg_mask, manip_seg_mask
    
    
def clean_mask_cup(demo_data):
    seg_mask = process_seg_mask_nav_manip(demo_data)
    left_eef_traj = np.array(demo_data['obs/prop_eef_state'][:, 13:20]) # num_steps, 7
    left_gripper_position = np.array(demo_data['obs/prop_eef_state'][:,20]) # num_steps, 7
    act_mask = remove_no_act_segment(left_eef_traj, left_gripper_position, seg_mask, window_size=5, tol=1e-3)
    
    return act_mask 


def clean_mask_tidy_table(demo_data):
    seg_mask = process_seg_mask_nav_manip_tidy_table(demo_data)
    left_eef_traj = np.array(demo_data['obs/prop_eef_state'][:, 13:20]) # num_steps, 7
    left_gripper_position = np.array(demo_data['obs/prop_eef_state'][:,20]) # num_steps, 7
    act_mask = remove_no_act_segment_tidy_table(left_eef_traj, left_gripper_position, seg_mask, window_size=5, tol=1e-3)
    
    return act_mask 

def clean_mask_clean_pan(demo_data):
    seg_mask = process_seg_mask_clean_pan(demo_data)
    left_eef_traj = np.array(demo_data['obs/prop_eef_state'][:, 13:20]) # num_steps, 7
    left_gripper_position = np.array(demo_data['obs/prop_eef_state'][:,20]) # num_steps, 7
    act_mask = remove_no_act_segment_clean_pan(left_eef_traj, left_gripper_position, seg_mask, window_size=5, tol=1e-3)
    
    return act_mask 
    
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
                # demo_data = raw_data[demo_id]
                
                if TABLE:
                    mask = clean_mask_tidy_table(demo_data)
                elif PAN:
                    mask = clean_mask_clean_pan(demo_data)
                else:
                    try:
                        mask = clean_mask_cup(demo_data)
                    except:
                        print("Error in cleaning mask", f"skipping demo {idx}")
                        continue
                
                # get the proprio data
                proprio_data = demo_data[STATE_KEY][:][mask]
                #get action
                raw_action = demo_data["actions"][:][mask] #first 3 base, 4 torso, 7 left arm, 7 right arm
                seq_length = proprio_data.shape[0]
                
                resized_images = {
                    key: resize_with_pad(
                            demo_data[key][...,:3][mask],
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
