import chess
import time
from backend.ai.random_ai import RandomAI
from backend.ai.material_ai import MaterialAI
from backend.ai.minimax_ai import MinimaxAI

board = chess.Board()
white_ai = MaterialAI()
black_ai = MinimaxAI(depth=3)  # Adjust depth as needed
move_count = 1

while not board.is_game_over():
  if board.turn == chess.WHITE:
    move = white_ai.select_move(board)
    board.push(move) if move else print("No legal moves available for White.")
  else:
    move = black_ai.select_move(board)
    board.push(move) if move else print("No legal moves available for Black.")


  # 最小限の表示（前の手と簡易盤面）
  print(f"\n[{move_count}]")
  print(board)
  time.sleep(0.25)
  move_count += 1

print("\nGame Over:", board.result())
