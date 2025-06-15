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
white_ai = AlphaBetaAI(depth=2)   # depth = 5 is max, but too slow
black_ai = NNUEAlphaBetaAI(depth=3)   # depth = 4 is max, but too slow

start_time = time.perf_counter()
clone_counter = hook_clone_counter()

while not board.is_game_over():
  turn = board.turn
  turn_start = time.perf_counter()

  if turn == chess.WHITE:
    player = white_ai
  else:
    player = black_ai

  move = player.select_move(board)
  board.push(move) if move else print(f"No legal moves available for {'White' if turn == chess.WHITE else 'Black'}.")

  print(f"\n[{move_count}]")
  print(board)

  ## DEBUG ##
  if turn == chess.BLACK:
    black_ai.evaluator.log_stats()
    black_ai.tt.log_stats()
    print(move, black_ai.best_score)

  turn_end = time.perf_counter()
  turn_execution_time = turn_end - turn_start
  print(f"{"White" if turn else "Black"} executed in: {turn_execution_time: .4f} seconds")
  move_count += 1

  time.sleep(0.25)

print("\nGame Over:", board.result())
end_time = time.perf_counter()
execution_time = end_time - start_time
 
print(f"Program executed in: {execution_time: .2f} seconds")
print(f"Tensor.clone() was called {clone_counter['clone']} times.")
