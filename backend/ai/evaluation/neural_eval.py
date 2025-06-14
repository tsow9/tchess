from backend.ai.nnue.nnue_model import SimpleNNUE
from backend.ai.utils.move_processing import apply_move_to_tensor
from backend.ai.utils.cache.evaluation_cache import EvalCache
from backend.ai.constants import local_paths
import torch
import chess
import os
import time

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
    self.cache = EvalCache()

    self.eval_count = 0
    self.total_eval_time = 0.0


  def __call__(self, board: chess.Board, tensor: torch.Tensor) -> float:
    start = time.perf_counter()

    if board.is_checkmate():
      score = -10000 if board.turn == chess.WHITE else 10000
    elif board.is_stalemate():
      score = 0.0

    with torch.no_grad():
      x = tensor.to(self.device).unsqueeze(0)
      score = self.model(x).item()

    elapsed = time.perf_counter() - start
    self.eval_count += 1
    self.total_eval_time += elapsed
    return score


  def log_stats(self):
    self.cache.log_stats()
    print(f"NN eval count: {self.eval_count}")
    print(f"Total eval time: {self.total_eval_time:.4f} sec")
    if self.eval_count > 0:
      print(f"Average time per eval: {self.total_eval_time / self.eval_count:.6f} sec")
