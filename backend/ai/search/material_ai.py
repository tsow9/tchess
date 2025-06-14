import chess
import random

from backend.ai.evaluation.material_eval import MaterialEvaluator

evaluate_board = MaterialEvaluator()


class MaterialAI:
  def select_move(self, board: chess.Board):
    best_move = None
    best_score = -float('inf') if board.turn == chess.WHITE else float('inf')

    # Shuffle legal moves to add some randomness
    legal_moves = list(board.legal_moves)
    random.shuffle(legal_moves)

    # Evaluate all legal moves
    for move in legal_moves:
      board.push(move)
      score = evaluate_board(board)
      board.pop()

      # Update best move based on the score
      if board.turn == chess.WHITE:
        if score > best_score:
          best_score = score
          best_move = move
      else:
        if score < best_score:
          best_score = score
          best_move = move

    return best_move
  



