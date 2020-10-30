import numpy as np
from PIL import Image
import time
import torch
from torchvision import transforms

resize = transforms.Compose(
    [transforms.Resize([128, 128], interpolation=Image.CUBIC,),],
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


def process_state(state):
    screen = np.ascontiguousarray(state, dtype=np.float32) / 255
    screen = torch.from_numpy(screen).permute(2, 0, 1)
    return resize(screen).unsqueeze(0)


def play_using_model(env, model, device, max_steps=10000):
    state = process_state(env.reset()).to(device)
    for step in range(max_steps):
        env.render()
        action = torch.argmax(model.forward(state), dim=1).item()
        state, reward, done, info = env.step(action)
        state = process_state(state).to(device)

        if done:
            break

        time.sleep(0.09)
