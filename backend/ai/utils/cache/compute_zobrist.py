import chess
from .zobrist import ZOBRIST_TABLE, ZOBRIST_CASTLING, ZOBRIST_EP, ZOBRIST_TURN

def compute_zobrist_hash(board: chess.Board) -> int:
  h = 0
  for square in chess.SQUARES:
    piece = board.piece_at(square)
    if piece is not None:
      piece_index = piece.piece_type - 1  # 0～5
      color_index = 0 if piece.color == chess.WHITE else 1
      h ^= ZOBRIST_TABLE[color_index][piece_index][square]

  # 手番
  if board.turn == chess.WHITE:
    h ^= ZOBRIST_TURN

  # キャスリング権
  if board.has_kingside_castling_rights(chess.WHITE):
    h ^= ZOBRIST_CASTLING[0]
  if board.has_queenside_castling_rights(chess.WHITE):
    h ^= ZOBRIST_CASTLING[1]
  if board.has_kingside_castling_rights(chess.BLACK):
    h ^= ZOBRIST_CASTLING[2]
  if board.has_queenside_castling_rights(chess.BLACK):
    h ^= ZOBRIST_CASTLING[3]

  # アンパッサン
  if board.ep_square is not None:
    file = chess.square_file(board.ep_square)
    h ^= ZOBRIST_EP[file]

  return h
