from backend.ai.nnue.nnue_model import SimpleNNUE
from backend.ai.utils.move_processing import apply_move_to_tensor
from backend.ai.constants import local_paths
import torch
import chess
import os

class NNUEEvaluator:
  def __init__(self):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model = SimpleNNUE().to(self.device)
    self.model.load_state_dict(
      torch.load(
        os.path.join(local_paths.DATA_MODELS, "model_D20250613_1605_AlphaBetaAI_d=2_vs_AlphaBetaAI_d=3_E80_B64.pth"), map_location=self.device
      )
    )
    self.model.eval()

  def __call__(self, board: chess.Board, tensor: torch.Tensor) -> float:
    if board.is_checkmate():
      return -10000 if board.turn == chess.WHITE else 10000
    elif board.is_stalemate():
      return 0.0

    with torch.no_grad():
      x = tensor.to(self.device).unsqueeze(0)
      return self.model(x).item()
