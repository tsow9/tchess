import chess
import random

from backend.ai.utils.cache.transposition_cache import TransPositionCache
from backend.ai.utils.cache.transposition_cache import TransPositionCacheEntry
from backend.ai.utils.cache.compute_zobrist import compute_zobrist_hash
from backend.ai.utils.conv_to_tensor import board_to_tensor
from backend.ai.utils.move_processing import apply_move_to_tensor
from backend.ai.utils.move_ordering import move_ordering_score
from backend.ai.evaluation.neural_eval import NNUEEvaluator


class NNUEAlphaBetaAI:
  def __init__(self, depth):
    self.depth = depth
    self.evaluator = NNUEEvaluator()
    self.cache = TransPositionCache()

    self.opening_books = {
      chess.WHITE: ["e2e4", "d2d4", "c2c4", "g1f3"],
      chess.BLACK: ["e7e5", "c7c5", "e7e6", "g8f6"]
    }


  def select_move(self, board):
    tensor = board_to_tensor(board)
    if board.fullmove_number == 1:
      move_uci = random.choice(self.opening_books[board.turn])
      return chess.Move.from_uci(move_uci)
    
    best_move = None
    alpha, beta = -float('inf'), float('inf')
    best_score = -float('inf') if board.turn == chess.WHITE else float('inf')
    is_maximizing = board.turn == chess.WHITE

    # If the evaluation are the same, try not to take the first possible move every time. 
    legal_moves = list(board.legal_moves)
    legal_moves.sort(key=lambda m: move_ordering_score(board, m), reverse=True)
    for move in legal_moves:
      new_tensor = apply_move_to_tensor(board, move, tensor)
      board.push(move)
      score = self.alphabeta(board, new_tensor, self.depth - 1, alpha, beta, not is_maximizing)
      board.pop()

      if board.turn == chess.WHITE and score > best_score:
        best_score = score
        best_move = move
        alpha = max(alpha, score)

      elif board.turn == chess.BLACK and score < best_score:
        best_score = score
        best_move = move
        beta = min(beta, score)

    print(self.evaluator.log_stats())
    print(self.cache.log_stats())

    return best_move
  

  def alphabeta(self, board, tensor, depth, alpha, beta, is_maximizing):
    board_hash_key = compute_zobrist_hash(board)
    alpha_org = alpha
    tt = self.cache.get(board_hash_key)
    if tt and tt.depth >= depth:
        if tt.flag == 'EXACT':
            return tt.score
        if tt.flag == 'LOWERBOUND' and tt.score >= beta:
            return tt.score
        if tt.flag == 'UPPERBOUND' and tt.score <= alpha:
            return tt.score

    # Calculate score for the current position
    if depth == 0 or board.is_game_over():
      # Check if the position is already evaluated
      cached_score = self.evaluator.cache.get(board_hash_key)
      if cached_score is not None:
        score = cached_score
      else:
        score = self.evaluator(board, tensor)
        self.evaluator.cache.set(board_hash_key, score)
      self.cache.set(board_hash_key, TransPositionCacheEntry(depth, score, 'EXACT'))
      return score

    # If the evaluation are the same, try not to take the first possible move every time. 
    max_score = -float('inf')
    min_score = float('inf')
    for move in sorted(board.legal_moves, key=lambda m: move_ordering_score(board, m), reverse=True):
      new_tensor = apply_move_to_tensor(board, move, tensor)
      board.push(move)
      score = self.alphabeta(board, new_tensor, depth - 1, alpha, beta, False)
      board.pop()
    
      if is_maximizing:
        max_score = max(max_score, score)
        alpha = max(alpha, score)
      else:
        min_score = min(min_score, score)
        beta = min(beta, score)
      if beta <= alpha:
        break
    
    score = max_score if is_maximizing else min_score
    if score <= alpha_org:
      flag = 'UPPERBOUND'
    elif score >= beta:
      flag = 'LOWERBOUND'
    else:
      flag = 'EXACT'

    self.cache.set(board_hash_key, TransPositionCacheEntry(depth, score ,flag))
    return score