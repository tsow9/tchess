import chess
import time
from backend.ai.search.random_ai import RandomAI
from backend.ai.search.material_ai import MaterialAI
from backend.ai.search.minimax_ai import MinimaxAI
from backend.ai.search.alphabeta_ai import AlphaBetaAI
from backend.ai.search.nnue_alphabeta_ai import NNUEAlphaBetaAI

from backend.ai.evaluation.material_eval import MaterialEvaluator
from backend.ai.evaluation.neural_eval import NNUEEvaluator

move_count = 1

board = chess.Board()
white_ai = AlphaBetaAI(depth=3)   # depth = 4 is max, but too slow
black_ai = NNUEAlphaBetaAI(depth=4)   # depth = 5 is max, but too slow


while not board.is_game_over():
  if board.turn == chess.WHITE:
    move = white_ai.select_move(board)
    board.push(move) if move else print("No legal moves available for White.")
  else:
    move = black_ai.select_move(board)
    board.push(move) if move else print("No legal moves available for Black.")

  print(f"\n[{move_count}]")
  print(board)
  time.sleep(0.25)
  move_count += 1

print("\nGame Over:", board.result())
