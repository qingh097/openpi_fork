import os 
import numpy as np
from openpi.training import config
from openpi.policies import policy_config
from openpi_client.image_tools import resize_with_pad

RESIZE_SIZE = 224

class OpenPIWrapper():
    def __init__(
        self, 
        model_ckpt_folder : str, 
        ckpt_id : int, 
        text_prompt : str = "put the white cup on the coffee machine",
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
        checkpoint_dir = os.path.join(model_ckpt_folder, f"{ckpt_id}")
        # Create a trained policy.
        self.policy = policy_config.create_trained_policy(config.get_config("pi0_fast_sim_b1k_450"), checkpoint_dir)
        self.text_prompt = text_prompt

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
        action = self.policy.infer(batch)
        # convert to absolute action and append gripper command
        # action["actions"] shape: (10, 21), joint_positions shape: (21,)
        # Need to broadcast joint_positions to match action sequence length
        target_joint_positions = action["actions"].copy()
        
        target_joint_positions[0] += joint_positions
        for i in range(1, target_joint_positions.shape[0]):
            target_joint_positions[i] += target_joint_positions[i-1]
        target_joint_positions[:,-8] = action["actions"][:,-8] # left gripper
        target_joint_positions[:,-1] = action["actions"][:,-1] # right gripper
        
        arms_action  = target_joint_positions[..., 7:]
        return {
            "mobile_base": target_joint_positions[..., :3],
            "torso": target_joint_positions[..., 3:7],
            "left_arm": arms_action[..., :6],
            "left_gripper": arms_action[..., 6:7],
            "right_arm": arms_action[..., 7:13],
            "right_gripper": arms_action[..., 13:14],
        }

