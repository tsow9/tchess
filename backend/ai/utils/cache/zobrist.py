import random

# Table for each square
ZOBRIST_TABLE = [[[random.getrandbits(64) for _ in range(64)] for _ in range(6)] for _ in range(2)]
# Which castling rights are there
ZOBRIST_CASTLING = [random.getrandbits(64) for _ in range(4)]
# Which en passant rights are there
ZOBRIST_EP = [random.getrandbits(64) for _ in range(8)]
# White or Black. If white, create one number out of 64
ZOBRIST_TURN = random.getrandbits(64)
