import chess
import random

from backend.ai.utils.conv_to_tensor import board_to_tensor
from backend.ai.utils.cache.compute_zobrist import compute_zobrist_hash
from backend.ai.utils.move_ordering import move_ordering_score
from backend.ai.utils.move_processing import apply_move_to_tensor
from backend.ai.utils.cache.transposition_table import TranspositionTable
from backend.ai.evaluation.neural_eval import NNUEEvaluator


class NNUEAlphaBetaAI:
  def __init__(self, depth):
    self.depth = depth
    self.evaluator = NNUEEvaluator()
    self.tt = TranspositionTable()
    self.best_score = 0

    self.opening_books = {
      chess.WHITE: ["e2e4", "d2d4", "c2c4", "g1f3"],
      chess.BLACK: ["e7e5", "c7c5", "e7e6", "g8f6"]
    }

  def select_move(self, board):
    tensor = board_to_tensor(board)
    delta = 50  # ウィンドウ幅（例：±50 centipawn）
    prev_score, best_move = 0, None

    if board.fullmove_number == 1:
      move_uci = random.choice(self.opening_books[board.turn])
      move = chess.Move.from_uci(move_uci)
      return move

    for depth in range(1, self.depth + 1):
      alpha = (-float('inf') if depth == 1 else prev_score - delta)
      beta  = ( float('inf') if depth == 1 else prev_score + delta)

      while True:
        score, move = self.search_root_depth(board, tensor, depth, alpha, beta)
        if score <= alpha:
          alpha = -float('inf')  # Fail-low -> re-exploration
        elif score >= beta:
          beta =  float('inf')  # Fail-high -> re-exploration
        else:
            break 

        delta *= 2

      prev_score, best_move = score, move

    self.best_score = prev_score
    
    return best_move



  def search_root_depth(self, board, tensor, depth, alpha, beta):
    moves, boards, tensors = [], [], []

    # Move ordering
    for move in board.legal_moves:
      new_tensor = apply_move_to_tensor(board, move, tensor)
      board.push(move)
      moves.append(move)
      boards.append(board.copy())
      tensors.append(new_tensor)
      board.pop()

    # {Zobrish_hash: score}
    score_dict = self.evaluator.evaluate_batch(boards, tensors)
    scored = list(zip(moves, boards, tensors))
    scored.sort(key=lambda mbt: score_dict[compute_zobrist_hash(mbt[1])],
                reverse=board.turn == chess.WHITE)

    # Search root loop
    best_move = None
    alpha, beta = -float('inf'), float('inf')
    best_score = -float('inf') if board.turn == chess.WHITE else float('inf')
    is_maximizing = board.turn == chess.WHITE

    for move, nbrd, ntensor in scored:
      score = self.alphabeta(nbrd, ntensor, depth - 1, alpha, beta, not is_maximizing)
      if board.turn == chess.WHITE and score > best_score:
        best_score = score
        best_move = move
        alpha = max(alpha, score)

      elif board.turn == chess.BLACK and score < best_score:
        best_score = score
        best_move = move
        beta = min(beta, score)

    return best_score, best_move
  


  def alphabeta(self, board, tensor, depth, alpha, beta, is_maximizing):
    zobrist_hash = compute_zobrist_hash(board)
    entry = self.tt.get(zobrist_hash)
    if entry and entry.depth >= depth:
      if entry.flag == "EXACT":
        return entry.score
      if entry.flag == "LOWER" and entry.score >= beta:
        return entry.score
      if entry.flag == "UPPER" and entry.score <= alpha:
        return entry.score

    # Calculate score for the current position
    if depth == 0 or board.is_game_over():
      score = self.evaluator(board, tensor)   # Using cache
      return score

    original_alpha, original_beta = alpha, beta
    best_move = None
    best_score = -float('inf') if is_maximizing else float('inf')

    target_moves = list(board.legal_moves)

    if entry and entry.move in target_moves:
      target_moves.remove(entry.move)
      target_moves.insert(0, entry.move)

    # Move ordering
    target_moves.sort(key=lambda m: move_ordering_score(board, m), reverse=True)
    if is_maximizing: 
      for move in target_moves:
        new_tensor = apply_move_to_tensor(board, move, tensor)
        board.push(move)
        score = self.alphabeta(board, new_tensor, depth - 1, alpha, beta, False)
        board.pop()
        if score > best_score:
          best_score = score
          best_move = move
        alpha = max(alpha, score)
        if beta <= alpha:
          break
    
    else:
      for move in target_moves:
        new_tensor = apply_move_to_tensor(board, move, tensor)
        board.push(move)
        score = self.alphabeta(board, new_tensor, depth - 1, alpha, beta, True)
        board.pop()
        if score < best_score:
          best_score = score
          best_move = move
        beta = min(beta, score)
        if beta <= alpha:
          break
    
    # Either alpha is cut, beta is cut, or not be cut
    if best_score <= original_alpha:
      flag = "UPPER"
    elif best_score >= original_beta:
      flag = "LOWER"
    else:
      flag = "EXACT"

    self.tt.set(zobrist_hash, depth, flag, score, best_move)
    return best_score