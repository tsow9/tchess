import random

class RandomAI:
    def select_move(self, board):
        legal_moves = list(board.legal_moves)
        return random.choice(legal_moves)
