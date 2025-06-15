import chess
import torch

def board_to_tensor(board: chess.Board) -> torch.Tensor:
  """
  Convert a chess.Board object to a tensor representation for neural evaluation.
  """
  # 64 squares, 6 piece types (pawn, knight, bishop, rook, queen, king),
  # 1 color (white and black), 4 castling rights, 8 en passtant files
  tensor = torch.zeros(64 * 6 * 2 + 1 + 4 + 8)   

  # 1. Piece positions
  piece_map = board.piece_map()
  for square, piece in piece_map.items():
    piece_type = piece.piece_type - 1  # 1~6 -> 0~5
    color_offset = 0 if piece.color == chess.WHITE else 6 * 64
    idx = color_offset + piece_type * 64 + square
    tensor[idx] = 1

  # 2. White or Black
  tensor[768] = 1 if board.turn == chess.WHITE else 0

  # 3. Castling rights
  tensor[769] = 1 if board.has_kingside_castling_rights(chess.WHITE) else 0
  tensor[770] = 1 if board.has_queenside_castling_rights(chess.WHITE) else 0
  tensor[771] = 1 if board.has_kingside_castling_rights(chess.BLACK) else 0
  tensor[772] = 1 if board.has_queenside_castling_rights(chess.BLACK) else 0

  # 4. En passant square
  if board.ep_square is not None:
    ep_file = chess.square_file(board.ep_square)  # 0 = a, ..., 7 = h
    tensor[773 + ep_file] = 1

  return tensor
