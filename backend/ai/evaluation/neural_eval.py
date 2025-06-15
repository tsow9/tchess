from backend.ai.nnue.nnue_model import SimpleNNUE
from backend.ai.utils.cache.compute_zobrist import compute_zobrist_hash
from backend.ai.utils.conv_to_tensor import board_to_tensor
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
        os.path.join(local_paths.DATA_MODELS, "model_Dcombined_20250614_2232_AlphaBetaAI_d=2_vs_AlphaBetaAI_d=2_E8_B64.pth"), map_location=self.device
      )
    )
    self.cache = EvalCache()

    self.eval_count = 0
    self.total_eval_time = 0.0


  def evaluate(self, board: chess.Board, tensor: torch.Tensor) -> float:
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
  
  def evaluate_batch(self, boards: list[chess.Board], tensors: list[torch.Tensor]) -> dict[int, float]:
    results = {}
    unc_board = []
    unc_zob_hash = []
    unc_tensors = []

    for board, tensor in zip(boards, tensors):
      zobrist_hash = compute_zobrist_hash(board)
      score = self.cache.get(zobrist_hash)
      if score is not None:
        results[zobrist_hash] = score
      else:
        unc_board.append(board)
        unc_zob_hash.append(zobrist_hash)
        unc_tensors.append(tensor)

    # If there is anything not cached, calculate the score
    if unc_board:
      x = torch.stack(unc_tensors).to(self.device)
      self.model.eval()
      with torch.no_grad():
        scores = self.model(x).squeeze(1).cpu().tolist()
      for zobrist_hash, score in zip(unc_zob_hash, scores):
        self.cache.set(zobrist_hash, score)
        results[zobrist_hash] = score

    return results

  

  
  def __call__(self, board: chess.Board, tensor: torch.Tensor):
    # If score has already calculated, it outputs cached score
    zobrist_hash = compute_zobrist_hash(board)
    cached_score = self.cache.get(zobrist_hash)
    if cached_score is not None:
      return cached_score
    # Score was not calculated, calculate score
    score = self.evaluate(board, tensor)
    self.cache.set(zobrist_hash, score)
    return score
  

  def log_stats(self):
    self.cache.log_stats()
    print(f"NN eval count: {self.eval_count}")
    print(f"Total eval time: {self.total_eval_time:.4f} sec")
    if self.eval_count > 0:
      print(f"Average time per eval: {self.total_eval_time / self.eval_count:.6f} sec")
    return 
