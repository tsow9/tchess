import hashlib
import chess
from backend.ai.utils.cache.compute_zobrist import compute_zobrist_hash

class TransPositionCache:
  def __init__(self):
    self.cache = {}

  def get(self, board: chess.Board):
    key = compute_zobrist_hash(board)  # Situation related Key. Same situation, same key
    return self.cache.get(key, None)   # Returns the score if the key is available, or None if not.

  def set(self, board: chess.Board, score: float):
    key = compute_zobrist_hash(board)   # Situation related Key. Same situation, same key
    self.cache[key] = score

  def clear(self):
    self.cache.clear()