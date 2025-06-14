import chess
import random

def move_ordering_score(board: chess.Board, move: chess.Move) -> int:
  """
  Change the order depending on the move you can make
  """
  score = 0

  # 1. promotion +1000
  if move.promotion:
    score += 1000

  # 2. The difference between the piece taken and the piece taken + 500
  if board.is_capture(move):
    captured_piece = board.piece_at(move.to_square)
    attacker = board.piece_at(move.from_square)
    if captured_piece and attacker:
      capture_score = piece_value(captured_piece.piece_type) - piece_value(attacker.piece_type)
      score += 500 + capture_score  # キャプチャは高優先

  # 3. Check +100
  temp_board = board.copy()
  temp_board.push(move)
  if temp_board.is_check():
    score += 100

  # 4. the other moves are 0 or 1. 
  ## When there are moves with the same score, it avoid choosing the same move every time.
  if score == 0:
    score += random.randint(0, 1)

  return score

def piece_value(piece_type: int) -> int:
  values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
  return values.get(piece_type, 0)
