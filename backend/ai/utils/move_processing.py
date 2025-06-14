import chess
import torch


def apply_move_to_tensor(board: chess.Board, move: chess.Move, tensor: torch.Tensor):
  tensor = tensor.clone()

  # 1. from_square to be 0 → to_square to be 1
  ## !Counting from the bottom left (a1)
  moving_piece = board.piece_at(move.from_square)
  if moving_piece is not None:
    piece_type = moving_piece.piece_type - 1
    color_offset = 0 if moving_piece.color == chess.WHITE else 6 * 64
    from_idx = color_offset + piece_type * 64 + move.from_square
    to_idx = color_offset + piece_type * 64 + move.to_square

    tensor[from_idx] = 0
    tensor[to_idx] = 1

    # 2. pawn's from_square to be 0, pawn's to_square including promotion to be 1
    if move.promotion is not None:
      piece_type = chess.PAWN - 1
      color_offset = 0 if moving_piece.color == chess.WHITE else 6 * 64
      from_idx = color_offset + piece_type * 64 + move.from_square
      promoted_type = move.promotion - 1
      to_idx = color_offset + promoted_type * 64 + move.to_square

      tensor[from_idx] = 0
      tensor[to_idx] = 1

  # 3. Remove the piece taken by the move
  # If there is another piece in the position where the piece is going, set that piece to 0
  if board.is_capture(move):   # If the move has taken another piece
    captured_piece = board.piece_at(move.to_square)
    if captured_piece is not None:
      cap_type = captured_piece.piece_type - 1
      cap_offset = 0 if captured_piece.color == chess.WHITE else 6 * 64
      cap_idx = cap_offset + cap_type * 64 + move.to_square
      tensor[cap_idx] = 0


  # 4. White or Black
  tensor[768] = 1 if board.turn == chess.WHITE else 0

  # 5. Castling rights
  tensor[769] = 1 if board.has_kingside_castling_rights(chess.WHITE) else 0
  tensor[770] = 1 if board.has_queenside_castling_rights(chess.WHITE) else 0
  tensor[771] = 1 if board.has_kingside_castling_rights(chess.BLACK) else 0
  tensor[772] = 1 if board.has_queenside_castling_rights(chess.BLACK) else 0

  # 6. En passant square
  tensor[773] = 1 if board.ep_square is not None else 0

  return tensor
