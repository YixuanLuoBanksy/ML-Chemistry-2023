import torch.nn as nn
import torch


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=4,kernel_size=5,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=16,stride=1,padding=1),
            nn.Conv1d(in_channels=4, out_channels=1, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=16, stride=1, padding=1),
            nn.Flatten(),
            nn.Linear(3471, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.layers(x)
        return x