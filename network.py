import torch
import torch.nn as nn

class Qnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(6, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.Tanh()
        )

    def forward(self, state):
        q_values = self.linear(state)
        return q_values
    
    def update(self):
        pass
