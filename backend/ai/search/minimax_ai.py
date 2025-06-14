import chess
import random
from backend.ai.evaluation.material_eval import MaterialEvaluator


class MinimaxAI:
  def __init__(self, depth):
    self.depth = depth
    self.evaluate_board = MaterialEvaluator()

  def select_move(self, board):
    best_move = None
    best_score = -float('inf') if board.turn == chess.WHITE else float('inf')

    legal_moves = list(board.legal_moves)
    random.shuffle(legal_moves)

    for move in legal_moves:
      board.push(move)
      score = self.minimax(board, self.depth - 1, not board.turn)
      board.pop()

      if board.turn == chess.WHITE and score > best_score:
        best_score = score
        best_move = move
      elif board.turn == chess.BLACK and score < best_score:
        best_score = score
        best_move = move

    return best_move

  def minimax(self, board, depth, is_maximizing):
    if depth == 0 or board.is_game_over():
      return self.evaluate_board(board)  # Material evaluation

    best = -float('inf') if is_maximizing else float('inf')
    
    legal_moves = list(board.legal_moves)
    random.shuffle(legal_moves)
    for move in legal_moves:
      # If this ai is white, Here field think of it in terms of depth, like black-white-black-white...
      board.push(move)
      score = self.minimax(board, depth - 1, not is_maximizing)  
      board.pop()
      best = max(best, score) if is_maximizing else min(best, score)

    return best
