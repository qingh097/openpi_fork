import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi_client.image_tools import resize_with_pad
import jax
import jax.numpy as jnp

MAX_DEPTH = 5.0
def make_ft_example() -> dict:
    """Creates a random input example for the Droid policy."""
    return {
        "observation/egocentric_camera": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_right": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(21),
        "prompt": "do something",
    }

def resize_with_pad_jax(image, target_h, target_w):
    h, w = image.shape[:2]
    scale = jnp.minimum(target_h / h, target_w / w)
    new_h = jnp.round(h * scale).astype(int)
    new_w = jnp.round(w * scale).astype(int)
    image = jax.image.resize(image, (new_h, new_w, image.shape[-1]), method="bilinear")

    pad_h = target_h - new_h
    pad_w = target_w - new_w
    pad_top = pad_h // 2
    pad_left = pad_w // 2
    image = jnp.pad(
        image,
        ((pad_top, pad_h - pad_top), (pad_left, pad_w - pad_left), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    return image

def _parse_image_jax(image):
    image = jnp.asarray(image)

    if jnp.issubdtype(image.dtype, jnp.floating):
        image = jnp.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image = jnp.clip(image, 0.0, MAX_DEPTH) / MAX_DEPTH
        image = (255.0 * image).astype(jnp.uint8)
        if image.shape[0] == 3:
            image = einops.rearrange(image, "c h w -> h w c")

    # ⚠️ resize_with_pad must also be jax-compatible
    image = resize_with_pad_jax(image, 224, 224)
    return image


def _parse_image(image):
    image = np.asarray(image)
    
    if np.issubdtype(image.dtype, np.floating):
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image = np.clip(image, 0.0, MAX_DEPTH) / MAX_DEPTH
        image = (255 * image).astype(np.uint8)
        if image.shape[0] == 3:
            image = einops.rearrange(image, "c h w -> h w c")
    image = resize_with_pad(
                            image,
                            224,
                            224
                        ) 
    return image

def safe_asarray(x):
    if isinstance(x, jax.Array):
        return jnp.asarray(x)
    else:
        return np.asarray(x)
    
@dataclasses.dataclass(frozen=True)
class FtInputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:

        proprio_state = data["observation/joint_position"]
        human_reference = data.get("human_reference", None)
        if human_reference is not None:
            human_reference = safe_asarray(human_reference).reshape(-1,18)[::4].flatten()
            if isinstance(proprio_state, jax.Array):
                state = jnp.concatenate([proprio_state, human_reference], axis=-1)
            else:
                state = np.concatenate([proprio_state, human_reference], axis=-1)
        else:
            state = proprio_state
        # state = transforms.pad_to_dim(state, self.action_dim)
        if "actions" in data:
            action =  data["actions"]
            # action = transforms.pad_to_dim(action, self.action_dim)

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        if isinstance(proprio_state, jax.Array):
            wrist_image_left = _parse_image_jax(data["observation/wrist_image_left"])
            wrist_image_right = _parse_image_jax(data["observation/wrist_image_right"])
        else:
            wrist_image_left = _parse_image(data["observation/wrist_image_left"])
            wrist_image_right = _parse_image(data["observation/wrist_image_right"])
        if isinstance(proprio_state, jax.Array):
            base_image = jnp.zeros_like(wrist_image_left) # centric rgb is not available in ft
        else:
            base_image = np.zeros_like(wrist_image_left) # centric rgb is not available in ft

        match self.model_type:
            case _model.ModelType.PI0:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image_left, wrist_image_right)
                if isinstance(proprio_state, jax.Array):
                    image_masks = (jnp.bool_(False), jnp.bool_(True), jnp.bool_(True))
                else:
                    image_masks = (np.False_, np.True_, np.True_) # there is no centric rgb
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (base_image, wrist_image_left, wrist_image_right)
                if isinstance(proprio_state, jax.Array):
                    image_masks = (jnp.bool_(False), jnp.bool_(True), jnp.bool_(True))
                else:
                    image_masks = (np.False_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = action

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class FtOutputs(transforms.DataTransformFn):
    action_dim: int = 54
    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims.
        return {"actions": np.asarray(data["actions"][..., :self.action_dim])}
