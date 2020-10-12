from torch import nn

from .utils import settings_is_valid


class DQNet(nn.Module):
    def __init__(self, settings):
        required_settings = ["epsilon", "gamma", "num_actions", "replay_size"]
        if not settings_is_valid(settings, required_settings):
            raise Exception(
                f"Settings object {settings} missing some required settings."
            )

        super(DQNet, self).__init__()

        self.num_actions = settings["num_actions"]
        self.epsilon = settings["epsilon"]
        self.gamma = settings["gamma"]
        self.replay_size = settings["replay_size"]

        # input size: N, 4, 250, 160
        self.conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=32,
            kernel_size=(14, 12),
            stride=(7, 6),
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        # input size: N, 32, 35, 26
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 4), stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        # input size: N, 64, 16, 12
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=0)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU(inplace=True)
        # input size: N, 64, 7, 4
        self.fc4 = nn.Linear(1792, 512)
        self.relu4 = nn.ReLU(inplace=True)
        # input size: N, 512
        self.fc5 = nn.Linear(512, self.num_actions)
        self.log_softmax5 = nn.LogSoftmax()
        # output size: N, num_actions

    def forward(self, x):
        # Layer 1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        # Layer 2
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        # Layer 3
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        # Layer 4 (flatten before passing to fc)
        out = self.fc4(out.view(out.size()[0], -1))
        out = self.relu4(out)
        # Layer 5
        out = self.fc5(out)
        out = self.log_softmax5(out)
        return out
