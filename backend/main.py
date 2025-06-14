import chess
import time
from backend.ai.search.random_ai import RandomAI
from backend.ai.search.material_ai import MaterialAI
from backend.ai.search.minimax_ai import MinimaxAI
from backend.ai.search.alphabeta_ai import AlphaBetaAI
from backend.ai.search.nnue_alphabeta_ai import NNUEAlphaBetaAI

from backend.ai.evaluation.material_eval import MaterialEvaluator
from backend.ai.evaluation.neural_eval import NNUEEvaluator
from .ai.utils.clone_counter import hook_clone_counter

move_count = 1

board = chess.Board()
white_ai = AlphaBetaAI(depth=3)   # depth = 5 is max, but too slow
black_ai = NNUEAlphaBetaAI(depth=3)   # depth = 4 is max, but too slow

start_time = time.perf_counter()
clone_counter = hook_clone_counter()

while not board.is_game_over():
  if board.turn == chess.WHITE:
    white_start_time = time.perf_counter()

    move = white_ai.select_move(board)
    board.push(move) if move else print("No legal moves available for White.")

    white_end_time = time.perf_counter()
    white_execution_time = white_end_time - white_start_time
    print(f"White executed in: {white_execution_time: .4f} seconds")
  else:
    black_start_time = time.perf_counter()

    move = black_ai.select_move(board)
    board.push(move) if move else print("No legal moves available for Black.")

    black_end_time = time.perf_counter()
    black_execution_time = black_end_time - black_start_time
    print(f"Black executed in: {black_execution_time: .4f} seconds")

  print(f"\n[{move_count}]")
  print(board)
  time.sleep(0.25)
  move_count += 1

print("\nGame Over:", board.result())
end_time = time.perf_counter()
execution_time = end_time - start_time
 
print(f"Program executed in: {execution_time: .2f} seconds")
print(f"Tensor.clone() was called {clone_counter['clone']} times.")
