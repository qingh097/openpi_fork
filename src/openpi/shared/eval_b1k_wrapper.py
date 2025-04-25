import numpy as np
from openpi_client.image_tools import resize_with_pad
from collections import deque
from openpi_client import websocket_client_policy as _websocket_client_policy
import logging

RESIZE_SIZE = 224

class OpenPIWrapper():
    def __init__(
        self, 
        host, 
        port, 
        text_prompt : str = "put the white cup on the coffee machine",
        control_mode : str = "temporal_ensemble",
    ) -> None:
        """
        Args:
            model_ckpt_folder: str, path to the model checkpoint folder
            ckpt_id: int, checkpoint id
            device: str, device to run the model on
            text_prompt: str, text prompt to use for the model
        Example:
        model_ckpt_folder = "/home/mfu/research/openpi/checkpoints/pi0_fast_yumi/pi0_fast_yumi_finetune"
        ckpt_id = 29999
        device = "cuda"
        """
        # Create a trained policy.
        self.policy = _websocket_client_policy.WebsocketClientPolicy(
            host=host,
            port=port,
        )
        logging.info(f"Server metadata: {self.policy.get_server_metadata()}")
        self.text_prompt = text_prompt
        self.control_mode = control_mode
        self.action_queue = deque([],maxlen=10)
        self.last_action = np.zeros((10, 21), dtype=np.float64)
        self.max_len = 8
    
    def reset(self):
        self.action_queue = deque([],maxlen=10)

    def act(self, nbatch):
        # TODO reformat data into the correct format for the model
        # TODO: communicate with justin that we are using numpy to pass the data. Also we are passing in uint8 for images 
        """
        Model input expected: 
            📌 Key: observation/exterior_image_1_left
            Type: ndarray
            Dtype: uint8
            Shape: (224, 224, 3)

            📌 Key: observation/exterior_image_2_left
            Type: ndarray
            Dtype: uint8
            Shape: (224, 224, 3)

            📌 Key: observation/joint_position
            Type: ndarray
            Dtype: float64
            Shape: (16,)

            📌 Key: prompt
            Type: str
            Value: do something
        
        Model will output:
            📌 Key: actions
            Type: ndarray
            Dtype: float64
            Shape: (10, 16)
        """
        
        if self.control_mode == 'receeding_horizon':
            if len(self.action_queue) > 0:
                # pop the first action in the queue
                final_action = self.action_queue.popleft()[None]
                arms_action  = final_action[..., 7:]
                
                return {
                    "mobile_base": final_action[..., :3],
                    "torso": final_action[..., 3:7],
                    "left_arm": arms_action[..., :6],
                    "left_gripper": arms_action[..., 6:7],
                    "right_arm": arms_action[..., 7:13],
                    "right_gripper": arms_action[..., 13:14],
                }
            
        # update nbatch observation (B, T, num_cameras, H, W, C) -> (B, num_cameras, H, W, C)
        nbatch["observation"] = nbatch["observation"][:, -1] # only use the last observation step
        if nbatch["observation"].shape[-1] != 3:
            # make B, num_cameras, H, W, C  from B, num_cameras, C, H, W
            # permute if pytorch
            nbatch["observation"] = np.transpose(nbatch["observation"], (0, 1, 3, 4, 2))

        # nbatch["proprio"] is B, T, 16, where B=1
        joint_positions = nbatch["proprio"][0, -1]
        batch = {
            "observation/egocentric_camera": resize_with_pad(
                nbatch["observation"][0, 0], 
                RESIZE_SIZE,
                RESIZE_SIZE
            ),
            "observation/wrist_image_left": resize_with_pad(
                nbatch["observation"][0, 1], 
                RESIZE_SIZE,
                RESIZE_SIZE
            ),
            "observation/wrist_image_right": resize_with_pad(
                nbatch["observation"][0, 2], 
                RESIZE_SIZE,
                RESIZE_SIZE
            ),
            "observation/joint_position": joint_positions,
            "prompt": self.text_prompt,
        }
        try:
            action = self.policy.infer(batch)
            self.last_action = action
        except:
            action = self.last_action
        # convert to absolute action and append gripper command
        # action["actions"] shape: (10, 21), joint_positions shape: (21,)
        # Need to broadcast joint_positions to match action sequence length
        target_joint_positions = action["actions"].copy()
        
        # target_joint_positions[0] += joint_positions
        # for i in range(1, target_joint_positions.shape[0]):
        #     target_joint_positions[i] += target_joint_positions[i-1]
        # target_joint_positions[:,-8] = action["actions"][:,-8] # left gripper
        # target_joint_positions[:,-1] = action["actions"][:,-1] # right gripper
        if self.control_mode == 'receeding_horizon':
            self.action_queue = deque([a for a in target_joint_positions[:self.max_len]])
            final_action = self.action_queue.popleft()[None]

        # # temporal emsemble start
        elif self.control_mode == 'temporal_ensemble':
            new_actions = deque(target_joint_positions)
            self.action_queue.append(new_actions)
            actions_current_timestep = np.empty((len(self.action_queue), target_joint_positions.shape[1]))
            
            # k = 0.01
            k = 0.005
            for i, q in enumerate(self.action_queue):
                actions_current_timestep[i] = q.popleft()

            exp_weights = np.exp(k * np.arange(actions_current_timestep.shape[0]))
            exp_weights = exp_weights / exp_weights.sum()

            final_action = (actions_current_timestep * exp_weights[:, None]).sum(axis=0)
            final_action[-8] = target_joint_positions[-8]
            final_action[-1] = target_joint_positions[-1]
            final_action = final_action[None]
        else:
            final_action = target_joint_positions
            
        arms_action  = final_action[..., 7:]
                
        return {
            "mobile_base": final_action[..., :3],
            "torso": final_action[..., 3:7],
            "left_arm": arms_action[..., :6],
            "left_gripper": arms_action[..., 6:7],
            "right_arm": arms_action[..., 7:13],
            "right_gripper": arms_action[..., 13:14],
        }