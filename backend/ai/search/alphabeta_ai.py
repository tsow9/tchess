import chess
import random
from backend.ai.evaluation.material_eval import MaterialEvaluator


class AlphaBetaAI:
  def __init__(self, depth):
    self.depth = depth
    self.evaluate_board = MaterialEvaluator()

    self.opening_books = {
      chess.WHITE: ["e2e4", "d2d4", "c2c4", "g1f3"],
      chess.BLACK: ["e7e5", "c7c5", "e7e6", "g8f6"]
    }


  def select_move(self, board):
    if board.fullmove_number == 1:
      move_uci = random.choice(self.opening_books[board.turn])
      return chess.Move.from_uci(move_uci)
    
    # Initialize values
    best_move = None
    alpha, beta = -float('inf'), float('inf')
    best_score = -float('inf') if board.turn == chess.WHITE else float('inf')
    is_maximizing = board.turn == chess.WHITE

    # If the evaluation are the same, try not to take the first possible move every time. 
    legal_moves = list(board.legal_moves)
    random.shuffle(legal_moves)

    for move in legal_moves:
      board.push(move)
      score = self.alphabeta(board, self.depth - 1, alpha, beta, not is_maximizing)
      board.pop()

      if is_maximizing:
        if score > best_score:
          best_score = score
          best_move = move
        alpha = max(alpha, score)
      else:
        if score < best_score:
          best_score = score
          best_move = move
        beta = min(beta, score)

    return best_move
  

  def alphabeta(self, board, depth, alpha, beta, is_maximizing):
    # Calculate score for the current position
    if depth == 0 or board.is_game_over():
      return self.evaluate_board(board)

    # If the evaluation are the same, try not to take the first possible move every time. 
    legal_moves = list(board.legal_moves)
    random.shuffle(legal_moves)
    if is_maximizing: 
      max_score = -float('inf')
      for move in legal_moves:
        board.push(move)
        score = self.alphabeta(board, depth - 1, alpha, beta, False)
        board.pop()
        max_score = max(max_score, score)
        alpha = max(alpha, score)
        if beta <= alpha:
          break
      return max_score
    else:
      min_score = float('inf')
      for move in legal_moves:
        board.push(move)
        score = self.alphabeta(board, depth - 1, alpha, beta, True)
        board.pop()
        min_score = min(min_score, score)
        beta = min(beta, score)
        if beta <= alpha:
          break

    return min_score
