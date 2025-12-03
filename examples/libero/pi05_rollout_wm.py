import collections
import dataclasses
import json
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
import random
import robosuite.utils.transform_utils as T
import robosuite.macros as macros

import libero.libero.envs.bddl_utils as BDDLUtils
import h5py
import os
from robosuite.utils.binding_utils import MjSimState
import torch
import csv
import sys
sys.path.append(os.path.abspath("/svl/u/ravenh/lacwm/robot_world_models-raven-lam/projects/latent_action_models"))

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "10.79.12.59"
    port: int = 8000
    wm_host: str = "0.0.0.0"
    wm_port: int = 9100
    
    resize_size: int = 224
    # replan_steps: int = 5
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_object_unseen"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 1023  # Random Seed (for reproducibility)
    
    save_data: bool = False
    random_selected_action: bool = True

def random_initial_states(env, initial_states):
    #sample an array of 50,4 floats between 0 and 0.01
    state = MjSimState.from_flattened(initial_states[0], env.sim)
    qpos_shape = state.qpos.shape
    qpos = initial_states[:,1:1+qpos_shape[0]]
    random_pos = np.random.uniform(-0.05, 0.05, (initial_states.shape[0], qpos_shape[0]))
    new_qpos = qpos + random_pos
    initial_states[:,1:1+qpos_shape[0]] = new_qpos
    return initial_states

def generate_batched_input(element, batch_size, noise_scale=1):
    new_element = element.copy()
    noise = np.random.randn(batch_size, 50, 32) * noise_scale
    for k,v in new_element.items():
        if isinstance(v, str):
            continue
        else:
            new_element[k] = v[None].repeat(batch_size, axis=0)
    payload = {**new_element, "_noise": noise}
    return payload

def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_object_unseen":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    action_selector = _websocket_client_policy.WebsocketClientPolicy(args.wm_host, args.wm_port)
    
    SAVE_INIT_STATES = False
    LOAD_INIT_STATES = True
    data_save_folder_path = f"/viscam/projects/dexs2r/libero_init/{args.task_suite_name}/seed_{args.seed}/"
    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)
        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)
        
        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        initial_states = random_initial_states(env, initial_states)
        if SAVE_INIT_STATES:
            save_folder_path = f"/viscam/projects/dexs2r/libero_init/{task_suite.tasks[task_id].problem_folder}/seed_{args.seed}/"
            os.makedirs(save_folder_path, exist_ok=True)
            init_states_path = os.path.join(
                save_folder_path,
                task_suite.tasks[task_id].init_states_file,
            )
            torch.save(initial_states, init_states_path)  
            success_mask = []
            
        if LOAD_INIT_STATES:
            task_name = task_description.replace(" ","_")
            init_states_folder = f'/viscam/projects/dexs2r/libero_init/{task_suite.tasks[task_id].problem_folder}/'
            init_states_path = os.path.join(init_states_folder, f'{task_name}_all.pruned_init')
            expert_demo_info_csv_path = os.path.join(init_states_folder, f'init_state_path.csv')
            
            csv_reader = csv.reader(open(expert_demo_info_csv_path, 'r'))
            next(csv_reader)
            #read all seeds, and indexes from the csv file, which is the second column, and third column respectively
            all_seeds = []
            all_indexes = []
            for row in csv_reader:
                task, seed, index = row
                if task == task_name:
                    all_indexes.append(int(index))
                    all_seeds.append(int(seed))
            # init_states_folder = '/viscam/projects/dexs2r/libero_init/libero_object_unseen/seed_591'
            # init_states_path = os.path.join(init_states_folder, f'{task_description.replace(" ","_")}.pruned_init')
            
            initial_states = torch.load(init_states_path)
            predefined_index = np.arange(len(initial_states))
            # predefined_index = [23]
            print(f"predefined episodes: {predefined_index}")

        else:
            predefined_index = range(args.num_trials_per_task)
        
        if args.save_data:
            data_folder = data_save_folder_path
            os.makedirs(data_folder, exist_ok=True)
            hdf5_path = f'{data_folder}/{task_description.replace(" ","_")}.hdf5'
            h5py_f = h5py.File(hdf5_path, "a")
            
            if 'data' not in h5py_f:
                grp = h5py_f.create_group("data")
            else:
                grp = h5py_f["data"]

            grp.attrs["env_name"] = env.problem_name
            bddl_file_name = env.env.bddl_file_name
            problem_info = BDDLUtils.get_problem_info(bddl_file_name)
            grp.attrs["problem_info"] = json.dumps(problem_info)
            grp.attrs["macros_image_convention"] = macros.IMAGE_CONVENTION
            grp.attrs["bddl_file_name"] = str(bddl_file_name.relative_to('/svl/u/ravenh/lacwm/LIBERO/'))
            grp.attrs["bddl_file_content"] = open(bddl_file_name, "r").read()
            
        # Start episodes
        task_episodes, task_successes = 0, 0
        # for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
        for episode_idx in tqdm.tqdm(predefined_index):
            logging.info(f"\nTask: {task_description}")
            
            expert_demo_data_path = os.path.join(init_states_folder, f'seed_{all_seeds[episode_idx]}', f'{task_name}.hdf5')
            expert_demo_data = h5py.File(expert_demo_data_path, 'r')['data']
            all_demo_keys = list(expert_demo_data.keys())
            cur_demo_key = all_demo_keys[episode_idx]
            cur_demo_data = expert_demo_data[cur_demo_key]
            goal_images = cur_demo_data['obs/agentview_rgb']
            
            model_xml = env.sim.model.get_xml()

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])


            # Setup
            t = 0
            plan_idx = args.replan_steps
            replay_images = []
            
            ###
            ee_states = []
            gripper_states = []
            joint_states = []
            robot_states = []
            agentview_images = []
            eye_in_hand_images = []
            actions = []
            states = []
            rewards = []
            dones = []
            ####

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                # try:
                if True:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue
                    
                    state_playback = env.sim.get_state().flatten()
                    states.append(state_playback)
                    gripper_states.append(obs["robot0_gripper_qpos"])
                    joint_states.append(obs["robot0_joint_pos"])
                    ee_states.append(
                        np.hstack(
                            (
                                obs["robot0_eef_pos"],
                                T.quat2axisangle(obs["robot0_eef_quat"]),
                            )
                        )
                    )
                    robot_states.append(env.env.get_robot_state_vector(obs))
                    agentview_images.append(obs["agentview_image"])
                    eye_in_hand_images.append(obs["robot0_eye_in_hand_image"])

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        payload = generate_batched_input(element, batch_size=20, noise_scale=1)
                        all_action_chunks = np.array(client.infer(payload)["actions"])
                        
                        if args.random_selected_action:
                            action_chunk = all_action_chunks[np.random.randint(0, len(all_action_chunks))]
                        else:
                            wm_obs = {
                                "action_chunks": all_action_chunks,
                                "agent_obs": img,
                                "wrist_obs": wrist_img,
                                "goal_image": goal_images[plan_idx],
                                'prediction_steps': 20,
                            }
                            print("selecting actions chunks......")
                            results = action_selector.infer(wm_obs)
                            plan_idx += args.replan_steps
                            plan_idx = min(plan_idx, len(goal_images)-1)
                            action_chunk = results['action']
                            print("selected action chunk index: ", results['index'])
                        
                        
                        # Query model to get action
                        # action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    actions.append(action)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                # except Exception as e:
                #     logging.error(f"Caught exception: {e}")
                #     break
                
            if args.save_data:
                #save to hdf5 file
                dones = np.zeros(len(actions)).astype(np.uint8)
                dones[-1] = 1 if done else 0
                rewards = np.zeros(len(actions)).astype(np.uint8)
                rewards[-1] = 1 if done else 0
                
                #delte demo_{episode_idx} if it exists
                if f"demo_{episode_idx}" in grp:
                    del grp[f"demo_{episode_idx}"]
                
                ep_data_grp = grp.create_group(f"demo_{episode_idx}")
                obs_grp = ep_data_grp.create_group("obs")
                obs_grp.create_dataset(
                    "gripper_states", data=np.stack(gripper_states, axis=0)
                )
                obs_grp.create_dataset("joint_states", data=np.stack(joint_states, axis=0))
                obs_grp.create_dataset("ee_states", data=np.stack(ee_states, axis=0))
                obs_grp.create_dataset("ee_pos", data=np.stack(ee_states, axis=0)[:, :3])
                obs_grp.create_dataset("ee_ori", data=np.stack(ee_states, axis=0)[:, 3:])

                obs_grp.create_dataset("agentview_rgb", data=np.stack(agentview_images, axis=0))
                obs_grp.create_dataset(
                    "eye_in_hand_rgb", data=np.stack(eye_in_hand_images, axis=0)
                )

                ep_data_grp.create_dataset("actions", data=actions)
                ep_data_grp.create_dataset("states", data=states)
                ep_data_grp.create_dataset("robot_states", data=np.stack(robot_states, axis=0))
                ep_data_grp.create_dataset("rewards", data=rewards)
                ep_data_grp.create_dataset("dones", data=dones)
                ep_data_grp.attrs["num_samples"] = len(agentview_images)
                ep_data_grp.attrs["model_file"] = model_xml
                ep_data_grp.attrs["init_state"] = states[0]
                ep_data_grp.attrs["success"] = True if done else False

            task_episodes += 1
            total_episodes += 1
            
            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}_{episode_idx}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            if SAVE_INIT_STATES:
                success_mask.append(bool(done))
                #save to json file
                with open(os.path.join(save_folder_path, f"success_mask_{task_segment}.json"), "w") as f:
                    json.dump(success_mask, f)
            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
        
        if args.save_data:
            grp.attrs["num_demos"] = task_episodes
            grp.attrs["total"] = task_episodes
            env.close()

            h5py_f.close()
        
        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
