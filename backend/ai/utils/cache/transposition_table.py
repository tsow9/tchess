from typing import Optional

class TTEntry:
  def __init__(self, zobrist_hash, depth, flag, score, move):
    self.zobrist_hash = zobrist_hash
    self.depth = depth
    self.flag = flag  # "EXACT", "LOWER", "UPPER"
    self.score = score
    self.move = move

class TranspositionTable:
  def __init__(self, size=1<<20):
    self.size = size   # 2^20
    self.table : list[Optional[TTEntry]] = [None] * size
    self.hit = self.miss = 0

  def idx(self, zobrist_hash):
    # Keep zobrist_hash in the range 0 to size-1 (bit mask)
    # Calculate in binary and use AND
    return zobrist_hash & (self.size - 1)

  def get(self, zobrist_hash):
    entry = self.table[self.idx(zobrist_hash)]
    if entry and entry.zobrist_hash == zobrist_hash:
      self.hit += 1
      return entry
    self.miss += 1
    return None

  def set(self, zobrist_hash, depth, flag, score, move):
    entry = TTEntry(zobrist_hash, depth, flag, score, move)
    self.table[self.idx(zobrist_hash)] = entry

  def log_stats(self):
    total = self.hit + self.miss
    hit_rate = self.hit / total * 100 if total > 0 else 0
    print(f"TranspositionTable hits: {self.hit}, misses: {self.miss}, hit rate: {hit_rate:.2f}%")