import chess

class MaterialEvaluator:
  PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0  # King is not capturable
  }

  def __call__(self, board: chess.Board) -> int:
    if board.is_checkmate():
      return -10000 if board.turn == chess.WHITE else 10000
    elif board.is_stalemate():
      return 0

    score = 0
    for square, piece in board.piece_map().items():
      value = self.PIECE_VALUES[piece.piece_type]
      score += value if piece.color == chess.WHITE else -value

    return score
