import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(
        self,
        d_model=512,
        d_ff=2048
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.network(x)