import numpy as np
from PIL import Image
import torch
from torchvision import transforms

resize = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(160, interpolation=Image.CUBIC),
        transforms.ToTensor(),
    ]
)


def settings_is_valid(settings, required_settings):
    """
    Simple check to determine whether provided
    settings object contains all the settings
    specified in required_settings.

    Inputs:
    - settings: dict
    - required_settings: list<str>
    """
    settings_set = set(settings.keys())
    required_set = set(required_settings)
    return len(settings_set.intersection(required_set)) == len(required_set)


def get_screen(env):
    screen = env.render(return_rgb_array=True).transpose((2, 0, 1))
    # Convert to float, rescale, convert to torch tensor
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    return resize(screen).unsqueeze(0)
