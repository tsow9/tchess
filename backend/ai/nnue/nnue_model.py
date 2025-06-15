import torch
import torch.nn as nn
import chess


class SimpleNNUE(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
      nn.Linear(64 * 6 * 2 + 1 + 4 + 8, 512),
      nn.ReLU(),
      nn.Linear(512, 128),
      nn.ReLU(),
      nn.Linear(128, 1)
    )

  def forward(self, x):
    return self.model(x)
  

