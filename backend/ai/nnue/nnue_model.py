import torch
import torch.nn as nn
import chess


class SimpleNNUE(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
      nn.Linear(64 * 6 * 2 + 6, 128),
      nn.ReLU(),
      nn.Linear(128, 1)
    )

  def forward(self, x):
    return self.model(x)
  

