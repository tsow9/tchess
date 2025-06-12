import chess

PIECE_VALUES = {
  chess.PAWN: 1,
  chess.KNIGHT: 3,
  chess.BISHOP: 3,
  chess.ROOK: 5,
  chess.QUEEN: 9,
  chess.KING: 0
}

def evaluate(board: chess.Board) -> int:
  score = 0
  for piece in board.piece_map().values():
    value = PIECE_VALUES[piece.piece_type]
    if piece.color == chess.WHITE:
      score += value
    else:
      score -= value
  return score