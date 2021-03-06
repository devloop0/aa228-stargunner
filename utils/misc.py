import numpy as np
from PIL import Image
import torch
from torchvision import transforms

resize = transforms.Compose([transforms.Resize([84, 84], interpolation=Image.CUBIC,),],)


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


def process_state(state):
    screen = np.ascontiguousarray(state, dtype=np.float16) / 255
    screen = torch.from_numpy(screen).permute(2, 0, 1)
    return resize(screen).unsqueeze(0).type(torch.float16)


def save_model_checkpoint(model, optimizer, timestep, out_filename):
    torch.save(
        {
            "timestep": timestep,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        out_filename,
    )


def save_model(model, out_filename):
    torch.save(model.state_dict(), out_filename)
