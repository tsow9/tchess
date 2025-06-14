import torch
import chess
import os
from datetime import datetime
import time
from backend.ai.utils.conv_to_tensor import board_to_tensor
from backend.ai.evaluation.material_eval import MaterialEvaluator
from backend.ai.search.alphabeta_ai import AlphaBetaAI
from backend.ai.search.random_ai import RandomAI

from backend.ai.constants import local_paths




def ai_game(ai_white, ai_black):
  board = chess.Board()
  positions = []

  while not board.is_game_over():
    tensor = board_to_tensor(board)   # Convert the board to an offset type
    positions.append((tensor.clone(), board.turn))

    move = ai_white.select_move(board) if board.turn == chess.WHITE else ai_black.select_move(board)
    if move is None:
      print("No legal moves available, game over.")
    else:
      board.push(move)


  result = board.result()  # '1-0', '0-1', '1/2-1/2'
  print("Game Over: ", result, "\n")

  result = board.result()  # '1-0', '0-1', '1/2-1/2'
  if result == "1-0":
    reward_white, reward_black = 10, 0.0
  elif result == "0-1":
    reward_white, reward_black = 0.0, 10
  else:
    reward_white = reward_black = 0

  data = []
  for tensor, turn in positions:
    reward = reward_white if turn == chess.WHITE else reward_black
    data.append((tensor, torch.tensor([reward])))

  return data


if __name__ == "__main__":

  material_evaluate = MaterialEvaluator()  # Use the material evaluation method

  round = 200

  ai_1_depth = 2
  ai_2_depth = 3
  ai_1 = AlphaBetaAI(depth=ai_1_depth)  # Use None for random AI
  ai_2 = AlphaBetaAI(depth=ai_2_depth)

  ai_1_name = ai_1.__class__.__name__
  ai_1_name += f"_d={ai_1_depth}" if ai_1_depth else ""
  ai_2_name = ai_2.__class__.__name__
  ai_2_name += f"_d={ai_2_depth}" if ai_1_depth else ""

  battle_name = f"{ai_1_name}_vs_{ai_2_name}"

  time.sleep(2)

  all_data = []
  print(f"Starting self-learning: {battle_name} with {round * 2} rounds...")
  for i in range(round):
    print(f"AIs first round 1/2: {i+1}/{round}")
    data = ai_game(ai_1, ai_2)
    all_data.extend(data)

  for i in range(round):
    print(f"AIs Second round 2/2: {i+1}/{round}")
    data = ai_game(ai_2, ai_1)
    all_data.extend(data)

  timestamp = datetime.now().strftime("%Y%m%d_%H%M")
  torch.save(all_data, os.path.join(local_paths.DATA_SELF_LEARNING_DATASET, f"{timestamp}_{battle_name}.pt"))
  print("Self-learning data saved!")
